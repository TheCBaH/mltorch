(* What an artifact may depend on at run time. The default adds no third-party
   math runtime: a helper whose native binding is a C library symbol is refused
   until the project owns an implementation. The system mode declares that
   dependency, and an image built under it carries it in its manifest. *)

module Art = Machine_model.Mir_artifact

type t = Dependency_free | System_libm

let name = function
  | Dependency_free -> "dependency_free"
  | System_libm -> "system_libm"

(* The C library symbols the artifact calls, in symbol order. *)
let helpers artifact =
  List.filter_map
    (fun (s : Art.Symbol.t) ->
      match s.Art.Symbol.kind with
      | Art.Symbol.External_function f -> Some f
      | Art.Symbol.Data _ | Art.Symbol.Function _ -> None)
    (Art.symbols artifact)

let admit mode artifact =
  match (mode, helpers artifact) with
  | _, [] | System_libm, _ -> Ok ()
  | Dependency_free, h :: _ -> Error (Rivet_a64_refusal.Helper h)
