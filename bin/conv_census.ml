(* Convolution census: a model's Conv2d and Linear nodes grouped by class, with
   node counts, multiply-adds and their share of the total. Classes follow how
   a kernel is chosen: pointwise (1x1), depthwise (groups = input channels),
   grouped, stem (at most four input channels) and dense KxK, each split by
   stride. Multiply-adds are output cells times the reduction extent; a
   padded window's clipped taps are not subtracted, so they are an upper bound.

   argv: <model.pt2> *)

let int_of_dim = Core.Dim.to_int

module Key = struct
  type t = { cls : string; kernel : int * int; stride : int * int }

  let compare = compare
end

module M = Map.Make (Key)

let () =
  match List.tl (Array.to_list Sys.argv) with
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
          prerr_endline "conv census: could not open or lower the archive";
          exit 1
      | Ok graph ->
          let tensors = graph.Graph_ir.Graph.tensors in
          let shape id =
            (Graph_ir.Tensor_id.Map.find id tensors).Tensor_sig.shape
          in
          let cells id = int_of_dim (Vec6.numel (shape id)) in
          let extent id axis = int_of_dim (Vec6.get (shape id) axis) in
          let table = ref M.empty and total = ref 0 in
          let add key macs =
            let n, m = Option.value (M.find_opt key !table) ~default:(0, 0) in
            table := M.add key (n + 1, m + macs) !table;
            total := !total + macs
          in
          List.iter
            (fun (node : Graph_ir.node) ->
              match (node.Graph_ir.Node.op, node.Graph_ir.Node.outputs) with
              | Graph_ir.Conv2d c, out :: _ ->
                  let p = c.Conv.Conv2d.params in
                  let h = p.Conv.Conv2d.h and w = p.Conv.Conv2d.w in
                  let kh = int_of_dim h.Conv.Conv2d.kernel
                  and kw = int_of_dim w.Conv.Conv2d.kernel in
                  let sh = (h.Conv.Conv2d.stride :> int)
                  and sw = (w.Conv.Conv2d.stride :> int) in
                  let groups = (p.Conv.Conv2d.groups :> int) in
                  let cin = int_of_dim p.Conv.Conv2d.in_channels in
                  let per_group = cin / groups in
                  let cls =
                    if groups = cin && groups > 1 then "depthwise"
                    else if groups > 1 then "grouped"
                    else if kh = 1 && kw = 1 then "pointwise"
                    else if cin <= 4 then "stem"
                    else "dense"
                  in
                  add
                    { Key.cls; kernel = (kh, kw); stride = (sh, sw) }
                    (cells out * per_group * kh * kw)
              | Graph_ir.Linear l, out :: _ ->
                  let k = extent l.Linear.Linear.weight Axis.C in
                  add
                    { Key.cls = "linear"; kernel = (1, 1); stride = (1, 1) }
                    (cells out * k)
              | _ -> ())
            (Graph_ir.nodes graph);
          Fmt.pr "%s: %d multiply-adds in convolutions and linears@."
            (Filename.basename path) !total;
          M.iter
            (fun (k : Key.t) (n, macs) ->
              Fmt.pr "  %-9s k%dx%d s%dx%d  %4d nodes  %12d  %5.1f%%@." k.cls
                (fst k.kernel) (snd k.kernel) (fst k.stride) (snd k.stride) n
                macs
                (100. *. float_of_int macs /. float_of_int (max 1 !total)))
            !table)
  | _ ->
      prerr_endline "usage: conv_census <model.pt2>";
      exit 2
