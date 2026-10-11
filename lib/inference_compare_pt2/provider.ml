(* An engine registered by a library above this one, selected by a variant
   word: this library names no compiler, loader or runtime, so a provider says
   whether its prerequisites are present and, if so, builds its runner. The
   engine identity is resolved when the runner is made, before any body is
   prepared, and a missing prerequisite is a refusal naming it, never a skip or
   another engine. *)

open Inference_compare_schema

type instance = {
  runner : Graph_ir.op Graph_body.t Graph_runner.t;
  identity : Engine_identity.t;
}

type t = {
  name : string;
  available : unit -> (unit, string) result;
  instantiate : unit -> (instance, string) result;
}

let table : (string, t) Hashtbl.t = Hashtbl.create 8
let register t = Hashtbl.replace table t.name t
let find name = Hashtbl.find_opt table name

let names () =
  List.sort String.compare (Hashtbl.fold (fun k _ acc -> k :: acc) table [])

(* The instance of a registered provider, or the reason it cannot be had. *)
let instantiate name =
  match find name with
  | None -> Error (Printf.sprintf "no engine %S is registered" name)
  | Some p -> (
      match p.available () with
      | Error m -> Error (Printf.sprintf "%s is not available: %s" name m)
      | Ok () -> p.instantiate ())
