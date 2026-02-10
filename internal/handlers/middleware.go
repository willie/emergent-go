package handlers

import (
	"log/slog"
	"net/http"
	"time"
)

// statusWriter wraps ResponseWriter to capture the status code
type statusWriter struct {
	http.ResponseWriter
	status int
	wrote  bool
}

func (w *statusWriter) WriteHeader(status int) {
	if !w.wrote {
		w.status = status
		w.wrote = true
		w.ResponseWriter.WriteHeader(status)
	}
}

func (w *statusWriter) Write(b []byte) (int, error) {
	if !w.wrote {
		w.status = 200
		w.wrote = true
	}
	return w.ResponseWriter.Write(b)
}

// Flush passes through to the underlying ResponseWriter if it supports flushing (required for SSE)
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// LogRequest is structured logging middleware using slog
func LogRequest(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}

		next.ServeHTTP(sw, r)

		attrs := []slog.Attr{
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", sw.status),
			slog.Duration("duration", time.Since(start)),
		}

		// Include session ID if present
		if cookie, err := r.Cookie(sessionCookieName); err == nil {
			cookiePreview := cookie.Value
			if len(cookiePreview) > 8 {
				cookiePreview = cookiePreview[:8]
			}
			attrs = append(attrs, slog.String("session", cookiePreview+"..."))
		}

		slog.LogAttrs(r.Context(), slog.LevelInfo, "request", attrs...)
	})
}
