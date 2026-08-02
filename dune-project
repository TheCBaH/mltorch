(lang dune 3.20)
; 3.20 is the floor for `(using melange 1.0)` -- dune rejects that extension
; below it, and the `(lang ...)` stanza has to stay on line 1. See
; .ai/js_backends_design.md.
(using menhir 2.1)
(using directory-targets 0.1)
(using ctypes 0.3)
(using melange 1.0)
(cram enable)

(name mlorch)
