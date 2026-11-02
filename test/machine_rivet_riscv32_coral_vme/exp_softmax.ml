(* What the generated binary32 [exp] and softmax cost on the core, planned for
   the scalar unit or forced onto the vector one: cycles for the whole image
   and per element. The softmax kernel of the fused attention is built from
   the same [exp]. A 64x64 softmax is 4.2 million simulated cycles, minutes of
   simulation: the default suite takes 16x16 and the 64x64 sizes have a
   directory of their own. *)

module C = Machine_rivet_riscv32_coral.Rv32_mir_coral
module T = Machine_model_test.Model_test

let run config ~sizes =
  let fixture name f ~h ~w =
    ( name,
      fun () ->
        Native_test.Graph_fixtures.build name
          Graph_builder.(
            let* x =
              input ~shape:(Native_test.Graph_fixtures.s 1 1 1 1 h w) ()
            in
            f x) )
  in
  let fixtures ~h ~w =
    [
      fixture "exp" ~h ~w (fun x -> Graph_builder.exp x);
      fixture "softmax over C" ~h ~w (fun x ->
          Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x);
      fixture "softmax over W" ~h ~w (fun x ->
          Graph_builder.softmax { Reduce.Softmax.axis = Axis.W } x);
    ]
  in
  List.iter
    (fun (target_name, target) ->
      List.iter
        (fun (h, w) ->
          List.iter
            (fun (name, g) ->
              let cycles = ref [] in
              let on_run (o : C.Run.t) = cycles := o.C.Run.mcycle :: !cycles in
              let label = Fmt.str "%s %dx%d, %s" name h w target_name in
              T.check
                ~route:(C.route ~on_run ~label config)
                ~contractions:true
                ~pipeline:
                  (Ssa_backends.Pipeline.Planned
                     { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target })
                ~oracle:T.Fp32 ~same:T.same_portable ~calls:1 label g;
              let total = List.fold_left Int64.add 0L !cycles in
              Fmt.pr "  cycles %Ld, %.1f per element@." total
                (Int64.to_float total /. float_of_int (h * w)))
            (fixtures ~h ~w))
        sizes)
    [
      ("planned", Ssa_ir.Ssa_target.zve32f128);
      ("forced vector", Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.zve32f128);
    ]
