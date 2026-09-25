// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

const chatCompletionOK = `{"id":"c1","object":"chat.completion","created":1,"model":"m",
	"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
	"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// keyServer answers each request by the key it carries: keys in limited get
// status, every other key gets ok. It records the key and body of every
// request in arrival order.
type keyServer struct {
	mu      sync.Mutex
	keys    []string
	bodies  []string
	limited map[string]bool
	status  int
	ok      string
	keyOf   func(*http.Request) string
}

func (s *keyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	key := s.keyOf(r)
	s.mu.Lock()
	s.keys = append(s.keys, key)
	s.bodies = append(s.bodies, string(body))
	s.mu.Unlock()
	if s.limited[key] {
		w.Header().Set("retry-after-ms", "1")
		w.WriteHeader(s.status)
		_, _ = w.Write([]byte(`{"error":{"message":"usage limit reached"}}`))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(s.ok))
}

func bearerKey(r *http.Request) string {
	return strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
}

func sendPing(t *testing.T, c LLMClient) error {
	t.Helper()
	_, err := c.CompletionsWithCtx(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "user", Content: "ping"}},
		MaxTokens: 16,
	})
	return err
}

func TestKeyFailover_SwitchesOnLimitAndStaysOnNextKey(t *testing.T) {
	for _, status := range []int{http.StatusTooManyRequests, http.StatusPaymentRequired} {
		srv := &keyServer{limited: map[string]bool{"k1": true}, status: status, ok: chatCompletionOK, keyOf: bearerKey}
		server := httptest.NewServer(srv)
		client := NewOpenAIClient(ClientConfig{URL: server.URL, APIKey: "k1", FallbackAPIKeys: []string{"k2"}, Model: "m"})

		if err := sendPing(t, client); err != nil {
			t.Fatalf("status %d: first request: %v", status, err)
		}
		if err := sendPing(t, client); err != nil {
			t.Fatalf("status %d: second request: %v", status, err)
		}
		server.Close()

		if got := strings.Join(srv.keys, ","); got != "k1,k2,k2" {
			t.Errorf("status %d: keys sent = %s, want k1,k2,k2 (switch once, then start on k2)", status, got)
		}
		if srv.bodies[0] == "" || srv.bodies[0] != srv.bodies[1] {
			t.Errorf("status %d: re-sent body differs from the original:\n%s\n%s", status, srv.bodies[0], srv.bodies[1])
		}
	}
}

func TestKeyFailover_AllKeysLimitedFallsBackToSDKRetries(t *testing.T) {
	srv := &keyServer{limited: map[string]bool{"k1": true, "k2": true}, status: http.StatusTooManyRequests, ok: chatCompletionOK, keyOf: bearerKey}
	server := httptest.NewServer(srv)
	defer server.Close()
	client := NewOpenAIClient(ClientConfig{URL: server.URL, APIKey: "k1", FallbackAPIKeys: []string{"k2"}, Model: "m"})

	if err := sendPing(t, client); err == nil {
		t.Fatal("expected an error when every key is limited")
	}
	var k1, k2 int
	for _, k := range srv.keys {
		switch k {
		case "k1":
			k1++
		case "k2":
			k2++
		}
	}
	// Failover must not multiply requests when the limit is not per key: the
	// SDK's own budget (1 attempt + 5 retries) bounds the total, and the keys
	// alternate across those attempts.
	if len(srv.keys) != 6 || k1 != 3 || k2 != 3 {
		t.Errorf("keys sent = %v, want 6 attempts alternating k1/k2", srv.keys)
	}
}

func TestKeyFailover_OtherErrorsDoNotSwitchKeys(t *testing.T) {
	srv := &keyServer{limited: map[string]bool{"k1": true}, status: http.StatusUnauthorized, ok: chatCompletionOK, keyOf: bearerKey}
	server := httptest.NewServer(srv)
	defer server.Close()
	client := NewOpenAIClient(ClientConfig{URL: server.URL, APIKey: "k1", FallbackAPIKeys: []string{"k2"}, Model: "m"})

	if err := sendPing(t, client); err == nil {
		t.Fatal("expected the 401 to surface")
	}
	for _, k := range srv.keys {
		if k != "k1" {
			t.Fatalf("keys sent = %v, want only k1: a rejected key is a config error, not a limit", srv.keys)
		}
	}
}

func TestKeyFailover_AnthropicUsesConfiguredHeader(t *testing.T) {
	const messageOK = `{"id":"msg","type":"message","role":"assistant","model":"m",
		"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
		"usage":{"input_tokens":1,"output_tokens":1}}`
	srv := &keyServer{
		limited: map[string]bool{"k1": true},
		status:  http.StatusTooManyRequests,
		ok:      messageOK,
		keyOf:   func(r *http.Request) string { return r.Header.Get("X-Api-Key") },
	}
	server := httptest.NewServer(srv)
	defer server.Close()
	client := NewAnthropicClient(ClientConfig{
		URL: server.URL + "/v1/messages", APIKey: "k1", FallbackAPIKeys: []string{"k2"}, Model: "m", AuthHeader: "x-api-key",
	})

	if err := sendPing(t, client); err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if got := strings.Join(srv.keys, ","); got != "k1,k2" {
		t.Errorf("X-Api-Key sent = %s, want k1,k2", got)
	}
}

func TestKeyRing_RotateKeepsConcurrentProgress(t *testing.T) {
	r := &keyRing{keys: []string{"a", "b", "c"}}
	if idx, _ := r.rotate(0); idx != 1 {
		t.Fatalf("rotate(0) = %d, want 1", idx)
	}
	// A second request that also saw key 0 limited must not skip key 1.
	if idx, key := r.rotate(0); idx != 1 || key != "b" {
		t.Errorf("stale rotate(0) = %d %q, want 1 \"b\"", idx, key)
	}
	r.rotate(1)
	if idx, _ := r.rotate(2); idx != 0 {
		t.Errorf("rotate past the last key = %d, want wrap to 0", idx)
	}
}
