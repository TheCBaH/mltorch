(* Shared rendering helpers for lower_test.ml and lower_shape_test.ml (split
   from a single lower_test.ml under the tracked file-size ceiling,
   scripts/check-file-size.sh). Not a test module itself. *)

open Native4d

(* Byte-identical to [Fixtures.build] and to verify_test.ml's former copy; one
   definition, in the module that already owns the fixtures. *)
let build = Fixtures.build

(* The result is packed over the destination version, so it cannot escape the
   scope that unpacks it — every consumer below renders inside. *)
let described ?constants g ~render =
  match Snapshot.create g with
  | Error e ->
      Format.asprintf "snapshot: %a" Graph_view.pp_error (Err.Error.kind e)
  | Ok (Snapshot.Pack src) -> (
      match Lower.convert ?constants src with
      | Error e -> Format.asprintf "%a" Error.pp (Err.Error.kind e)
      | Ok (Lower.Pack r) -> render (Lower.graph r))

(* Prints the destination graph, so a legalization producing the wrong op — or
   the wrong number of them — is visible rather than merely "ok". *)
let show name ?constants g =
  Format.printf "@[<v 2>%s:@,%s@]@." name
    (described ?constants g ~render:(Format.asprintf "%a" Graph.pp))

let outcome name ?constants g =
  Format.printf "%-26s %s@." name
    (described ?constants g ~render:(fun dst ->
         Format.asprintf "converted, %d nodes"
           (List.length dst.Graph_common.Graph.nodes)))
