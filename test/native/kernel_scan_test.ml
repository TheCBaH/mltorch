(* [Kernel.Limits.max_scan_updates_total]: the one scan resource dimension
   [Region_program.preflight] does not check, since it is Kernel-scoped -- the
   sum of [keys * per_key] over every value. Split out of kernel_test.ml to
   keep that file under the repository's file-size cap. *)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let f32 = Payload.Fmt Payload.F32
let tid = Tensor_id.of_int
let sg id shape = Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt:f32 ()

(* trace.(0,l) = 0; trace.(s+1,l) = trace.(s,l) + 1, [width=steps=1] -- its own
   [per_key] is 1. With a singleton partition every one of the value's
   [C=100] output cells is its own key, so the KERNEL total is 100:
   comfortably under a generous [max_scan_updates_total] and over a tight
   one, while [max_scan_updates_per_key] never moves. *)
let scan_value id shape =
  let limits =
    Err.or_raise ~pp_error:Expr.Scan_limits.pp_error
      (Expr.Scan_limits.create ~max_state:100 ~max_updates:1000L)
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scan ~limits ~width:1 ~steps:1
            ~init:(fun ~lane:_ -> Expr.Builder.return (Expr.Value.const 0.))
            ~update:(fun ~step:_ ~lane ~previous_at ->
              Expr.Builder.return
                (Expr.Value.add (previous_at lane) (Expr.Value.const 1.)))
            (fun scan_read ->
              Region_program.Builder.finish ~max_size:64 ~max_depth:16
                ~partition:Region_partition.singleton
                ~output:
                  (scan_read
                     ~row:(Expr.Index.clamp_low (Expr.Index.const 1))
                     ~lane:(Expr.Index.clamp_low (Expr.Index.const 0))))))
  in
  {
    Kernel.Value.id = tid id;
    sg = sg id shape;
    computation = program;
    result = Kernel.Result_conversion.Round_f32;
  }

let%expect_test
    "Kernel: max_scan_updates_total sums keys * per_key across values" =
  let limits ~max_scan_updates_total =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:4096 ~max_depth:128 ~max_values:16
         ~max_dep_depth:16 ~max_inputs:16 ~max_outputs:16
         ~max_extent:0x7FFF_FFFFL ~max_numel:0x7FFF_FFFFL ~max_local_slots:8192
         ~max_scan_state:8192 ~max_scan_updates_per_key:8192L
         ~max_scan_updates_total)
  in
  let show ~max_scan_updates_total =
    match
      Kernel.create
        ~limits:(limits ~max_scan_updates_total)
        ~inputs:[]
        ~values:[ scan_value 0 (s1c 100) ]
        ~outputs:[ tid 0 ]
        ()
    with
    | Ok _ -> "ok"
    | Error error -> Format.asprintf "%a" Kernel.pp_error (Err.Error.kind error)
  in
  Format.printf "generous=%s tight=%s@."
    (show ~max_scan_updates_total:200L)
    (show ~max_scan_updates_total:50L);
  [%expect
    {| generous=ok tight=summed scan updates across all values exceed limit 50 |}]
