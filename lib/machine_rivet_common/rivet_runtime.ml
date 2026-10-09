(* What an artifact may depend on at run time. The default adds no third-party
   math runtime: a helper whose native binding is a C library symbol is refused
   unless the project owns an implementation, which the image then carries.
   The system mode declares the C library as a dependency instead, binds every
   such helper to it, and an image built under it records that in its
   manifest. *)

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

(* The helpers the project implements itself ({!Machine_ir.Mir_exp}). *)
let owned_names = [ "exp" ]
let is_owned h = List.mem h owned_names

(* The first helper the mode forbids. *)
let admit mode artifact =
  match mode with
  | System_libm -> Ok ()
  | Dependency_free -> (
      match List.filter (fun h -> not (is_owned h)) (helpers artifact) with
      | [] -> Ok ()
      | h :: _ -> Error h)

(* The helpers the image carries its own code for, and those it binds to the
   C library: the two partition {!helpers}. *)
let carried mode artifact =
  match mode with
  | System_libm -> []
  | Dependency_free -> List.filter is_owned (helpers artifact)

let bound mode artifact =
  match mode with
  | System_libm -> helpers artifact
  | Dependency_free ->
      List.filter (fun h -> not (is_owned h)) (helpers artifact)
