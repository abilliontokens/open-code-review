// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"strings"
	"testing"
)

func TestResolveEndpoint_OpenCodeGoPresetHeaders(t *testing.T) {
	clearAllEnv(t)
	t.Setenv("OPENCODE_API_KEY", "")

	t.Run("preset session header applies by default", func(t *testing.T) {
		path, _ := writeResolverConfig(t, configFile{
			Provider:  "opencode-go",
			Providers: map[string]providerEntryConfig{"opencode-go": {APIKey: "sk-go", Model: "kimi-k3"}},
		})
		ep, err := ResolveEndpoint(path)
		if err != nil {
			t.Fatalf("ResolveEndpoint: %v", err)
		}
		if ep.URL != "https://opencode.ai/zen/go/v1" || ep.Protocol != ProtocolOpenAIChatCompletions {
			t.Errorf("endpoint = %s %s, want the Go chat completions endpoint", ep.Protocol, ep.URL)
		}
		if got := ep.ExtraHeaders["x-opencode-session"]; got != SessionKeyTemplateVar {
			t.Errorf("x-opencode-session = %q, want %q", got, SessionKeyTemplateVar)
		}
	})

	t.Run("entry headers override the preset and keep the rest", func(t *testing.T) {
		path, _ := writeResolverConfig(t, configFile{
			Provider: "opencode-go",
			Providers: map[string]providerEntryConfig{"opencode-go": {
				APIKey: "sk-go",
				Model:  "kimi-k3",
				ExtraHeaders: map[string]string{
					"x-opencode-session": "fixed",
					"x-team":             "review",
				},
			}},
		})
		ep, err := ResolveEndpoint(path)
		if err != nil {
			t.Fatalf("ResolveEndpoint: %v", err)
		}
		if ep.ExtraHeaders["x-opencode-session"] != "fixed" || ep.ExtraHeaders["x-team"] != "review" {
			t.Errorf("ExtraHeaders = %v, want entry values to win", ep.ExtraHeaders)
		}
		if p, _ := LookupProvider("opencode-go"); p.ExtraHeaders["x-opencode-session"] != SessionKeyTemplateVar {
			t.Error("resolving mutated the registry's preset headers")
		}
	})
}

func TestResolveEndpoint_APIKeysOrderAndPromotion(t *testing.T) {
	clearAllEnv(t)
	t.Setenv("OPENCODE_API_KEY", "sk-env")
	tests := []struct {
		name         string
		entry        providerEntryConfig
		wantToken    string
		wantFallback []string
	}{
		{"api_key first, then api_keys without repeats or blanks",
			providerEntryConfig{APIKey: "a", APIKeys: []string{"b", "a", "  ", "c", "b"}}, "a", []string{"b", "c"}},
		{"api_keys alone promotes its first key",
			providerEntryConfig{APIKeys: []string{"x", "y"}}, "x", []string{"y"}},
		{"a single api_keys entry has nothing to fail over to",
			providerEntryConfig{APIKeys: []string{"x"}}, "x", nil},
		{"static keys shadow the env var",
			providerEntryConfig{APIKeys: []string{" ", "x"}}, "x", nil},
		{"no static keys falls back to the env var",
			providerEntryConfig{APIKeys: []string{" "}}, "sk-env", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.entry.Model = "kimi-k3"
			path, _ := writeResolverConfig(t, configFile{
				Provider:  "opencode-go",
				Providers: map[string]providerEntryConfig{"opencode-go": tt.entry},
			})
			ep, err := ResolveEndpoint(path)
			if err != nil {
				t.Fatalf("ResolveEndpoint: %v", err)
			}
			if ep.Token != tt.wantToken || strings.Join(ep.FallbackTokens, ",") != strings.Join(tt.wantFallback, ",") {
				t.Errorf("keys = %q + %q, want %q + %q", ep.Token, ep.FallbackTokens, tt.wantToken, tt.wantFallback)
			}
		})
	}
}

func TestResolveEndpoint_APIKeysOnCustomProvider(t *testing.T) {
	clearAllEnv(t)
	path, _ := writeResolverConfig(t, configFile{
		Provider: "gw",
		CustomProviders: map[string]providerEntryConfig{"gw": {
			URL: "https://gw.example.com/v1", Protocol: "openai", Model: "m", APIKeys: []string{"k1", "k2"},
		}},
	})
	ep, err := ResolveEndpoint(path)
	if err != nil {
		t.Fatalf("ResolveEndpoint: %v", err)
	}
	if ep.Token != "k1" || len(ep.FallbackTokens) != 1 || ep.FallbackTokens[0] != "k2" {
		t.Errorf("keys = %q + %q, want k1 + [k2]", ep.Token, ep.FallbackTokens)
	}
}
