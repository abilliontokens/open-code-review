// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import "testing"

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
