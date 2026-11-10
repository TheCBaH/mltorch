(* Capacity census: for every invocation of a model's bundle, the bytes of the
   tensors it binds (each distinct edge once), against a region size. A slot is
   placed whole in one region ([Coral_layout.place]), so an invocation whose
   largest tensor exceeds the region cannot run in it however the rest is
   arranged; one whose total exceeds it cannot run with every tensor resident.
   Source tensors and outputs only: packed weight copies, scratch contractions
   add and the runtime stores are not counted, so a pass here is necessary, not
   sufficient.

   argv: <model.pt2> [--limit=BYTES] (default 4194304, the standalone cores'
   external RAM) *)

open Loop_ir

let bytes (sg : Tensor_sig.t) =
  let cells = (Vec6.numel sg.Tensor_sig.shape :> int) in
  cells * Payload.packed_cell_bytes sg.Tensor_sig.fmt

let () =
  let limit = ref 4_194_304 in
  let args =
    List.filter
      (fun a ->
        match String.split_on_char '=' a with
        | [ "--limit"; n ] ->
            limit := int_of_string n;
            false
        | _ -> true)
      (List.tl (Array.to_list Sys.argv))
  in
  match args with
  | [ path ] -> (
      let result =
        let open Err.Syntax in
        let lift e = (e :> [ Pt2_archive.error | Native_interp.error ]) in
        let* archive = Pt2_archive.open_pt2 path |> Err.map_error lift in
        let* lowered =
          Native_interp.lower_archive archive |> Err.map_error lift
        in
        Err.return lowered.Pt2_native_graph.graph
      in
      match Err.payload result with
      | Error _ ->
          prerr_endline "capacity census: could not open or lower the archive";
          exit 1
      | Ok graph -> (
          match Loop_bundle.build graph with
          | Error e ->
              Fmt.epr "capacity census: %a@." Loop_bundle.pp_error
                (Err.Error.kind e);
              exit 1
          | Ok b ->
              let tensors = b.Loop_bundle.graph.Graph_ir.Graph.tensors in
              let sig_bytes id =
                match Graph_ir.Tensor_id.Map.find_opt id tensors with
                | Some sg -> bytes sg
                | None -> 0
              in
              let over = ref 0 and worst_total = ref 0 and worst_one = ref 0 in
              List.iteri
                (fun i (inv : Loop_bundle.invocation) ->
                  let ids =
                    List.sort_uniq Graph_ir.Tensor_id.compare
                      inv.Loop_bundle.edges
                  in
                  let sizes = List.map sig_bytes ids in
                  let total = List.fold_left ( + ) 0 sizes in
                  let largest = List.fold_left max 0 sizes in
                  worst_total := max !worst_total total;
                  worst_one := max !worst_one largest;
                  if total > !limit then begin
                    incr over;
                    Fmt.pr
                      "invocation %d: %d bytes bound, largest tensor %d%s@." i
                      total largest
                      (if largest > !limit then " (alone over the limit)"
                       else "")
                  end)
                b.Loop_bundle.invocations;
              Fmt.pr
                "%s: %d invocations, %d over %d bytes; worst total %d, largest \
                 single tensor %d@."
                (Filename.basename path)
                (List.length b.Loop_bundle.invocations)
                !over !limit !worst_total !worst_one))
  | _ ->
      prerr_endline "usage: coral_capacity_census <model.pt2> [--limit=BYTES]";
      exit 2
