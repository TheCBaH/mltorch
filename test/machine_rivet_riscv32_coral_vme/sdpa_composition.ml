(* The per-op composition of an [Sdpa] on the core: the scores product
   granted the engine's pair rounding, then the scale, a softmax and a
   binary32 context product on the vector unit, bitwise against the
   descriptor's [Paired] scores followed by the binary32 oracle of the rest.
   The cycles are the bar the fused kernel must beat. The context product on
   the vector unit is about 97% of the cycles and grows about fourfold per
   doubling, so the 64x64 case is a run of minutes: it has a directory of its
   own, outside the default suite. *)

module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let run config ~tq ~tk ~d =
  let module K = Contraction in
  let flat (Tensor.Tensor t as p) =
    let a = Array.make (Vec6.numel t.Tensor.shape :> int) 0. in
    Vec6.iter t.Tensor.shape (fun c ->
        a.((Vec6.offset t.Tensor.shape c :> int)) <- Tensor.read p c);
    a
  in
  let name = Fmt.str "sdpa composition %dx%d d%d" tq tk d in
  let tail = T.sdpa_composition_tail_fixture ~tq ~tk ~d in
  let oracle =
    T.Custom
      (fun (g : Graph_ir.graph) _constants inputs ->
        let q, kt, v =
          match inputs with
          | [ (_, q); (_, kt); (_, v) ] -> (flat q, flat kt, v)
          | _ -> assert false
        in
        let desc =
          Result.get_ok
            (Err.payload
               (K.make ~batch:1L ~m:(Int64.of_int tq) ~n:(Int64.of_int tk)
                  ~k:(Int64.of_int d)
                  ~a:{ K.Operand.batch = 0L; outer = Int64.of_int d; k = 1L }
                  ~b:{ K.Operand.batch = 0L; outer = 1L; k = Int64.of_int tk }
                  ~c:{ K.Output.batch = 0L; row = Int64.of_int tk; col = 1L }
                  ~format:K.Element.Bf16 ~seed:K.Seed.Zero
                  ~numerics:K.Numerics.Paired))
        in
        let s = Array.make (tq * tk) 0. in
        K.eval desc
          ~a:(fun o -> q.(Int64.to_int o))
          ~b:(fun o -> kt.(Int64.to_int o))
          ~c:(fun _ -> 0.)
          ~write:(fun o x -> s.(Int64.to_int o) <- x);
        let tg = tail () in
        let tb = T.bundle tg in
        let s_id, v_id =
          match tb.Loop_ir.Loop_bundle.inputs with
          | [ a; b ] -> (a, b)
          | _ -> assert false
        in
        let s_shape = (T.sig_of tg s_id).Tensor_sig.shape in
        let s_t =
          Tensor.materialize s_shape (fun c ->
              s.((Vec6.offset s_shape c :> int)))
        in
        let out =
          T.expected T.Fp32 tg tb ~constants:[]
            ~inputs:[ (s_id, s_t); (v_id, v) ]
        in
        let tid = List.hd tg.Graph_ir.Graph.outputs
        and gid = List.hd g.Graph_ir.Graph.outputs in
        Tensor_id.Map.singleton gid (Tensor_id.Map.find tid out))
  in
  let cycles = ref [] in
  let on_run (o : C.Run.t) = cycles := o.C.Run.mcycle :: !cycles in
  T.check
    ~route:(C.route ~on_run ~label:name config)
    ~contractions:true ~bf16_pairs:true
    ~pipeline:
      (Ssa_backends.Pipeline.Planned
         {
           numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
           target = Ssa_ir.Ssa_target.zve32f128;
         })
    ~oracle ~same:T.same_portable
    ~input:(fun ~salt id sg -> T.in_format sg (T.values ~salt:(salt + id) sg))
    ~calls:1 name
    (T.sdpa_composition_bf16_fixture ~tq ~tk ~d);
  Fmt.pr "  cycles %Ld = %a@."
    (List.fold_left Int64.add 0L !cycles)
    Fmt.(list ~sep:(any " + ") int64)
    (List.rev !cycles)
