// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
)

// keyRing holds every API key configured for one provider and the key new
// requests start from. It is shared by all requests of a client, so once a key
// hits its usage limit the rest of the run moves on instead of rediscovering
// the limit on every request.
type keyRing struct {
	mu   sync.Mutex
	keys []string
	cur  int
}

func (r *keyRing) current() (int, string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cur, r.keys[r.cur]
}

// rotate moves past the key at from and returns the key now current. When a
// concurrent request already rotated away from it, the ring stays where that
// request left it rather than skipping a key nobody has tried yet.
func (r *keyRing) rotate(from int) (int, string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cur == from {
		r.cur = (from + 1) % len(r.keys)
	}
	return r.cur, r.keys[r.cur]
}

// isKeyLimitStatus reports whether a response says this key, rather than the
// request, cannot be served: 429 for rate and usage limits, 402 for an
// exhausted balance or quota.
func isKeyLimitStatus(code int) bool {
	return code == http.StatusTooManyRequests || code == http.StatusPaymentRequired
}

// keyFailoverFor returns the failover middleware for cfg, or nil when cfg
// carries a single key and there is nothing to fail over to.
func keyFailoverFor(cfg ClientConfig, authHeader string) func(*http.Request, func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	if len(cfg.FallbackAPIKeys) == 0 {
		return nil
	}
	keys := make([]string, 0, len(cfg.FallbackAPIKeys)+1)
	keys = append(keys, cfg.APIKey)
	keys = append(keys, cfg.FallbackAPIKeys...)
	return keyFailoverMiddleware(keys, authHeader)
}

// keyFailoverMiddleware sends each HTTP attempt with the ring's current key
// and, when the provider answers with a key limit status, re-sends it at once
// with the next key, trying each key at most once per attempt. When every key
// is limited the last response goes back to the SDK, whose own retry loop then
// backs off and comes through here again.
//
// It must be registered before every other middleware so that it is the
// outermost: each re-send then passes through the retry observer and the raw
// capture like any other attempt.
//
// authHeader is the header the key travels in: "authorization" (sent as a
// Bearer token) or the name of a header carrying the bare key.
func keyFailoverMiddleware(keys []string, authHeader string) func(*http.Request, func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	ring := &keyRing{keys: keys}
	return func(req *http.Request, next func(*http.Request) (*http.Response, error)) (*http.Response, error) {
		idx, key := ring.current()
		for tried := 1; ; tried++ {
			setAuthKey(req, authHeader, key)
			resp, err := next(req)
			if err != nil || !isKeyLimitStatus(resp.StatusCode) || tried >= len(keys) {
				return resp, err
			}
			retry := req.Clone(req.Context())
			if req.Body != nil && req.Body != http.NoBody {
				if req.GetBody == nil {
					return resp, err
				}
				body, bodyErr := req.GetBody()
				if bodyErr != nil {
					return resp, err
				}
				retry.Body = body
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()

			limited := idx
			idx, key = ring.rotate(idx)
			fmt.Fprintf(os.Stderr, "[ocr] WARNING: API key %d of %d hit a usage limit (HTTP %d); switching to key %d\n",
				limited+1, len(keys), resp.StatusCode, idx+1)
			req = retry
		}
	}
}

func setAuthKey(req *http.Request, authHeader, key string) {
	switch authHeader {
	case "", "authorization":
		req.Header.Set("Authorization", "Bearer "+key)
	default:
		req.Header.Set(authHeader, key)
	}
}
