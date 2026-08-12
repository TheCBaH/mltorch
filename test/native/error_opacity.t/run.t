Provenance: what the type checker must reject.

[Err.Error.t] pairs a payload with the callstack captured where the error was
detected. Every negative case below is a way of putting a payload back into a
wrapper by hand, which silently attaches the WRONG detection site. Each is
paired with a control that must still compile -- without the controls a broken
harness would reject everything and look like a pass. See CLAUDE.md.

  $ R=../../..
  $ check() { sh $R/test/native/error_opacity.sh $R/test/native/core_probe.exe "$1" "$2"; }

Record update is the dangerous one: it typechecks, reads like a widening, and
republishes the error under whatever backtrace [e] happened to carry.
[Err.map_error] is the supported spelling and keeps the original detection
site.

  $ check "rebuild by record update" "ignore { e with Err.Error.kind = 1 }"
  rebuild by record update: rejected
  $ check "widen via map_error" "ignore (Err.map_error succ r)"
  widen via map_error: COMPILES

Building a wrapper from nothing is the same defect without the disguise: the
backtrace names this line rather than the failure.

  $ check "forge a wrapper literal" "ignore { Err.Error.kind = 1; Err.Error.backtrace = Printexc.get_callstack 0 }"
  forge a wrapper literal: rejected
  $ check "build through Error.make" "ignore (Err.Error.make 1)"
  build through Error.make: COMPILES

Reading is fine and must stay fine -- opacity is about construction, not
inspection. Destructuring the record is rejected only because the field labels
are gone; the accessors give the same access with no way to put a value back.

  $ check "destructure the record" "match e with { Err.Error.kind; _ } -> ignore kind"
  destructure the record: rejected
  $ check "read the payload" "ignore (Err.Error.kind e)"
  read the payload: COMPILES
  $ check "read the provenance" "ignore (Err.Error.origin e)"
  read the provenance: COMPILES
