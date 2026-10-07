(* The named math helpers a lowering calls: binary64 [cos], [exp], [log] and
   [sin], each a versioned pure descriptor that cannot fail, bound to the C
   library function of the same name. A native realization declares and links
   that symbol; the interpreters' model is the host's binary64 libm — the
   primitive the SSA oracle and the C and Wasm backends already use — so
   agreement with them holds by construction, and another platform's libm
   differing is a documented disagreement, never a tolerance. Error function
   is not here: it is owned, expanded by the lowering into primitives and an
   [exp] call, in the oracle's own operation order. *)

module Fn = struct
  type t = Cos | Exp | Log | Sin

  let all = [ Cos; Exp; Log; Sin ]

  let name = function
    | Cos -> "cos"
    | Exp -> "exp"
    | Log -> "log"
    | Sin -> "sin"

  (* the helper id a program declares it under *)
  let id fn =
    Mir_id.Helper.of_int
      (match fn with Cos -> 0 | Exp -> 1 | Log -> 2 | Sin -> 3)
end

let version = 1

(* The C library symbol a native realization binds. *)
let libm_symbol = Fn.name

let descriptor fn =
  {
    Mir_helper.id = Fn.id fn;
    name = Fn.name fn;
    version;
    params = [ Mir_type.F64 ];
    results = [ Mir_type.F64 ];
    effects = Mir_helper.Effect.Pure;
    failures = [];
  }
