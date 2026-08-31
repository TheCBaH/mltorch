(* The Kernel reference interpreter, against the oracles it must reproduce.

   Four ways of computing the same thing, compared BITWISE ([Tensor.equal_bits],
   never [Float.equal], which equates -0. with 0. and any NaN with any other):

     Kernel_eval.run        buffer-based, the unfused path
     Kernel_eval.value_at   recursive, the on-demand path
     Stage_program.ground   the existing whole-stage oracle
     Eval_direct.run        the concrete front-end oracle

   That [run] and [value_at] agree is useful before any fusion exists: it is the
   test that fails if [value_at] forgets or doubles the result conversion, since
   only the converted reading equals the stored tensor. *)

open Graph_ir

type error =
  [ `Adapt of Kernel_adapt.error
  | `Build of Graph_builder.error
  | `Direct of Eval_direct.error
  | `Eval of Kernel_eval.error ]

let pp_error ppf : [< error ] -> unit = function
  | `Adapt e -> Kernel_adapt.pp_error ppf e
  | `Build e -> Graph_builder.pp_error ppf e
  | `Direct e -> Eval_direct.pp_error ppf e
  | `Eval e -> Kernel_eval.pp_error ppf e

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error
let lift f r = Err.map_error (fun e -> f e) r
let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n

(* Deterministic, backend-portable data: small integers, exactly representable
   in f32 and identical under js_of_ocaml. *)
let fill shape f = Tensor.materialize shape f

let ramp shape =
  fill shape (fun c -> float_of_int (Dim.to_int (Vec6.get c Axis.C) + 1))

(* Compare a kernel against all three oracles over one graph. *)
let compare_all ?inputs g =
  let open Err.Syntax in
  let prog = Eval_symbolic.run g in
  let* k = lift (fun e -> `Adapt e) (Kernel_adapt.of_stage_program prog) in
  let tensors =
    match inputs with
    | Some ts -> ts
    | None ->
        List.map
          (fun id ->
            (id, ramp (Tensor_id.Map.find id g.Graph.tensors).Tensor_sig.shape))
          g.Graph.inputs
  in
  let bind id = List.assoc_opt id tensors in
  let bind_total id = List.assoc id tensors in
  let* kernel = lift (fun e -> `Eval e) (Kernel_eval.run k ~bind) in
  (* [Eval_direct] separates the two source kinds; [Kernel_eval] does not — a
     [Caller] and a [Captured_constant] input both consult [bind], since the
     distinction is provenance, not how the data arrives. *)
  let is_constant id =
    match Tensor_id.Map.find_opt id g.Graph.input_kinds with
    | Some Input.Constant -> true
    | Some Input.Input | None -> false
  in
  let constants, plain =
    List.partition (fun (id, _) -> is_constant id) tensors
  in
  let* direct =
    lift (fun e -> `Direct e) (Eval_direct.run ~constants g ~inputs:plain)
  in
  let grounded = Stage_program.ground prog ~bind:bind_total in
  (* [value_at] at every coordinate of every value. *)
  let* recursive =
    List.fold_left
      (fun acc (v : Kernel.Value.t) ->
        let* m = acc in
        let cells = ref [] in
        let* () =
          Vec6.fold_coords v.Kernel.Value.sg.Tensor_sig.shape
            ~init:(Err.return ()) ~f:(fun acc c ->
              let* () = acc in
              let+ x =
                lift
                  (fun e -> `Eval e)
                  (Kernel_eval.value_at k ~bind v.Kernel.Value.id
                     (Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int c)))
              in
              cells := x :: !cells)
        in
        let cells = Array.of_list (List.rev !cells) in
        let i = ref 0 in
        let t =
          Tensor.materialize v.Kernel.Value.sg.Tensor_sig.shape (fun _ ->
              let x = cells.(!i) in
              incr i;
              x)
        in
        Err.return (Tensor_id.Map.add v.Kernel.Value.id t m))
      (Err.return Tensor_id.Map.empty)
      k.Kernel.values
  in
  let agree name other =
    ( name,
      Tensor_id.Map.for_all
        (fun id t ->
          match Tensor_id.Map.find_opt id other with
          | None -> false
          | Some u -> Tensor.equal_bits t u)
        kernel )
  in
  Err.return
    [
      agree "value_at" recursive;
      agree "ground" grounded;
      ( "direct",
        Tensor_id.Map.for_all
          (fun id t ->
            match Tensor_id.Map.find_opt id direct with
            | None ->
                true (* direct also holds inputs; values are the overlap *)
            | Some u -> Tensor.equal_bits t u)
          kernel );
    ]

let pp_agreements ppf rows =
  Fmt.list ~sep:Fmt.cut (fun ppf (n, ok) -> Fmt.pf ppf "%s: %b" n ok) ppf rows

let%expect_test "Kernel_eval: every fixture agrees with all three oracles" =
  let results =
    List.filter_map
      (fun (name, g) ->
        let g = g () in
        match compare_all g with
        | Ok rows when List.for_all snd rows -> None
        | Ok rows ->
            Some
              (Format.asprintf "%s: %a" name
                 (Fmt.list ~sep:(Fmt.any " ") (fun ppf (n, ok) ->
                      Fmt.pf ppf "%s=%b" n ok))
                 rows)
        | Error e ->
            Some
              (Format.asprintf "%s: %a" name
                 (Core.Pretty.error_kind pp_error)
                 e))
      Graph_fixtures.all
  in
  Format.printf "@[<v>fixtures: %d@,disagreeing or rejected: %d@,%a@]@."
    (List.length Graph_fixtures.all)
    (List.length results)
    (Fmt.list ~sep:Fmt.cut Fmt.string)
    results;
  [%expect
    {|
    fixtures: 45
    disagreeing or rejected: 2
    bypass_permute_mixed_compatibility: t3: a stored value must be f32 and unquantized, got f16
    reuse_permute_backtrack_candidate: t2: a stored value must be f32 and unquantized, got f16 |}]

let%expect_test "Kernel_eval: Region values execute through run and value_at" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let local_coord =
    Expr.Coord.make ~n:(Expr.Index.output Axis.N) ~t:(Expr.Index.output Axis.T)
      ~d:(Expr.Index.output Axis.D) ~h:(Expr.Index.output Axis.H)
      ~w:(Expr.Index.output Axis.W) ~c:Expr.Index.zero
  in
  let program =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar
            (Expr.Value.load
               (Expr_bridge.source_of_id (Tensor_id.of_int 0))
               local_coord)
            (fun local ->
              Region_program.Builder.finish ~max_size:32 ~max_depth:16
                ~partition
                ~output:
                  (Expr.Value.add local
                     (Expr.Value.load
                        (Expr_bridge.source_of_id (Tensor_id.of_int 0))
                        (Expr_bridge.coord_of_vec6 Symbolic.out_vec))))))
  in
  let input =
    {
      Kernel.Input.id = Tensor_id.of_int 0;
      sg =
        Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"" ~shape:(s1c 3)
          ~fmt:(Payload.Fmt Payload.F32) ();
      binding = Kernel.Binding.Caller;
    }
  in
  let value =
    {
      Kernel.Value.id = Tensor_id.of_int 1;
      sg =
        Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"" ~shape:(s1c 3)
          ~fmt:(Payload.Fmt Payload.F32) ();
      computation = program;
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  let kernel =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create ~inputs:[ input ] ~values:[ value ]
         ~outputs:[ Tensor_id.of_int 1 ]
         ())
  in
  let input_tensor =
    Tensor.materialize (s1c 3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C) + 1))
  in
  let bind id =
    if Tensor_id.equal id (Tensor_id.of_int 0) then Some input_tensor else None
  in
  let output =
    Tensor_id.Map.find (Tensor_id.of_int 1)
      (Err.or_raise ~pp_error:Kernel_eval.pp_error
         (Kernel_eval.run kernel ~bind))
  in
  let value_at =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.value_at kernel ~bind (Tensor_id.of_int 1)
         (Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2))
  in
  Fmt.pr "run: %g,%g,%g@.value_at: %g@."
    (Tensor.read output Vec6.origin)
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:1))
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2))
    value_at;
  [%expect {|
    run: 2,3,4
    value_at: 4 |}]

let%expect_test
    "Kernel_eval: Region consumers read stored Pixel producers under every \
     default path" =
  let partition =
    Err.or_raise ~pp_error:Region_partition.pp_error
      (Region_partition.of_whole_axes [ Axis.C ])
  in
  let local_coord =
    Expr.Coord.make ~n:(Expr.Index.output Axis.N) ~t:(Expr.Index.output Axis.T)
      ~d:(Expr.Index.output Axis.D) ~h:(Expr.Index.output Axis.H)
      ~w:(Expr.Index.output Axis.W) ~c:Expr.Index.zero
  in
  let source id coordinate =
    Expr.Value.load (Expr_bridge.source_of_id (Tensor_id.of_int id)) coordinate
  in
  let region =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_program.Builder.run
         (Region_program.Builder.scalar (source 1 local_coord) (fun local ->
              Region_program.Builder.finish ~max_size:32 ~max_depth:16
                ~partition
                ~output:
                  (Expr.Value.add local
                     (source 1 (Expr_bridge.coord_of_vec6 Symbolic.out_vec))))))
  in
  let sg id =
    Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape:(s1c 3)
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let kernel =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create
         ~inputs:
           [
             {
               Kernel.Input.id = Tensor_id.of_int 0;
               sg = sg 0;
               binding = Kernel.Binding.Caller;
             };
           ]
         ~values:
           [
             {
               Kernel.Value.id = Tensor_id.of_int 1;
               sg = sg 1;
               computation =
                 Region_program.pixel
                   (Expr.Value.add
                      (source 0 (Expr_bridge.coord_of_vec6 Symbolic.out_vec))
                      (Expr.Value.const 1.));
               result = Kernel.Result_conversion.Round_f32;
             };
             {
               Kernel.Value.id = Tensor_id.of_int 2;
               sg = sg 2;
               computation = region;
               result = Kernel.Result_conversion.Round_f32;
             };
           ]
         ~outputs:[ Tensor_id.of_int 2 ]
         ())
  in
  let input =
    Tensor.materialize (s1c 3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C) + 1))
  in
  let bind id =
    if Tensor_id.equal id (Tensor_id.of_int 0) then Some input else None
  in
  let run =
    Err.or_raise ~pp_error:Kernel_eval.pp_error (Kernel_eval.run kernel ~bind)
  in
  let planned =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run_plan (Fusion_plan.default kernel) ~bind)
  in
  let output = Tensor_id.Map.find (Tensor_id.of_int 2) run in
  let same =
    Tensor.equal_bits output (Tensor_id.Map.find (Tensor_id.of_int 2) planned)
  in
  let on_demand =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.value_at kernel ~bind (Tensor_id.of_int 2)
         (Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2))
  in
  Fmt.pr "run: %g,%g,%g@.default-plan: %b@.value_at: %g@."
    (Tensor.read output Vec6.origin)
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:1))
    (Tensor.read output (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:2))
    same on_demand;
  [%expect {|
    run: 4,5,6
    default-plan: true
    value_at: 6 |}]

let%expect_test "Kernel_eval: Pixel scan is C-innermost with one load per cell"
    =
  let sg id =
    Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape:(s1c 3)
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let body =
    Expr.Value.load
      (Expr_bridge.source_of_id (Tensor_id.of_int 0))
      (Expr_bridge.coord_of_vec6 Symbolic.out_vec)
  in
  let kernel =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create
         ~inputs:
           [
             {
               Kernel.Input.id = Tensor_id.of_int 0;
               sg = sg 0;
               binding = Kernel.Binding.Caller;
             };
           ]
         ~values:
           [
             {
               Kernel.Value.id = Tensor_id.of_int 1;
               sg = sg 1;
               computation = Region_program.pixel body;
               result = Kernel.Result_conversion.Round_f32;
             };
           ]
         ~outputs:[ Tensor_id.of_int 1 ]
         ())
  in
  let seen = ref [] in
  let input =
    Tensor.materialize (s1c 3) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.C) + 1))
  in
  ignore
    (Err.or_raise ~pp_error:Kernel_eval.pp_error
       (Kernel_eval.run
          ~on_load:(fun _ coord -> seen := coord.Expr.Coord.c :: !seen)
          kernel
          ~bind:(fun id ->
            if Tensor_id.equal id (Tensor_id.of_int 0) then Some input else None)));
  Fmt.pr "loads=%d C=%s@." (List.length !seen)
    (String.concat "," (List.map string_of_int (List.rev !seen)));
  [%expect {| loads=3 C=0,1,2 |}]

let build name outputs m =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    (Graph_builder.build ~name ~outputs m)

let%expect_test "Kernel_eval: conv, pool and reduce agree bitwise" =
  let conv_axis k st p : Conv.Conv2d.axis_window =
    {
      kernel = Dim.extent k;
      stride = Op_config.Pos.of_int st;
      pad_before = Op_config.Nonneg.of_int p;
      pad_after = Op_config.Nonneg.of_int p;
      dilation = Op_config.Pos.of_int 1;
    }
  in
  let graphs =
    [
      ( "conv2d",
        build "conv"
          (fun r -> [ r ])
          Graph_builder.(
            let* x = input ~shape:(s 1 1 1 3 3 2) () in
            let* w = input ~shape:(s 1 1 1 2 2 2) () in
            let* b = input ~shape:(s1c 1) () in
            conv2d
              {
                h = conv_axis 2 1 0;
                w = conv_axis 2 1 0;
                in_channels = Dim.extent 2;
                groups = Op_config.Pos.of_int 1;
              }
              ~x ~weight:w ~bias:b ()) );
      ( "max_pool_with_indices",
        build "pool"
          (fun (v, _) -> [ v ])
          Graph_builder.(
            let* x = input ~shape:(s 1 1 1 4 4 2) () in
            max_pool2d_with_indices
              {
                ceil_mode = false;
                kernel = { h = Dim.extent 2; w = Dim.extent 2 };
                stride =
                  { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
                pad =
                  {
                    h = Op_config.Nonneg.of_int 0;
                    w = Op_config.Nonneg.of_int 0;
                  };
              }
              x) );
      ( "mean",
        build "mean"
          (fun r -> [ r ])
          Graph_builder.(
            let* x = input ~shape:(s 1 1 1 2 3 4) () in
            mean { Reduce.Mean.dims = [ Axis.W ]; keepdim = true } x) );
    ]
  in
  List.iter
    (fun (name, g) ->
      Format.printf "@[<v>%s:@,%a@]@." name (pp_result pp_agreements)
        (compare_all g))
    graphs;
  [%expect
    {|
    conv2d:
    value_at: true
    ground: true
    direct: true
    max_pool_with_indices:
    value_at: true
    ground: true
    direct: true
    mean:
    value_at: true
    ground: true
    direct: true |}]

let%expect_test "Kernel_eval: the result conversion is applied exactly once" =
  (* The sweep above cannot test this. Its data is small integers, so every
     stage result is exactly representable in f32 and the round is a no-op —
     dropping it entirely would leave all four oracles agreeing.

     2^24+1 is the smallest positive integer f32 cannot hold, so this chain
     separates the three possibilities:

       once (correct):  2^24 -> +1 -> round = 2^24 -> +1 -> round = 2^24
       never:           2^24 -> +1 -> 2^24+1 -> +1 -> 2^24+2
       twice:           same as once TODAY, because Round_f32 is idempotent —
                        which is why the evaluator names eval_body and
                        eval_value apart rather than relying on that. *)
  let g =
    build "round_chain"
      (fun r -> [ r ])
      Graph_builder.(
        let* x = input ~shape:(s1c 1) () in
        let* a = add_scalar 1.0 x in
        add_scalar 1.0 a)
  in
  let inputs =
    [ (List.hd g.Graph.inputs, fill (s1c 1) (fun _ -> 16777216.0)) ]
  in
  Format.printf "@[<v>%a@]@." (pp_result pp_agreements) (compare_all ~inputs g);
  let k =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.of_stage_program (Eval_symbolic.run g))
  in
  let out =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run k ~bind:(fun id -> List.assoc_opt id inputs))
  in
  List.iter
    (fun (v : Kernel.Value.t) ->
      Format.printf "%a = %.1f@." Tensor_id.pp v.Kernel.Value.id
        (Tensor.read_at_raw (Tensor_id.Map.find v.Kernel.Value.id out) (fun _ ->
             0)))
    k.Kernel.values;
  Format.printf "without any round it would be %.1f@." (16777216.0 +. 1.0 +. 1.0);
  [%expect
    {|
    value_at: true
    ground: true
    direct: true
    t1 = 16777216.0
    t2 = 16777216.0
    without any round it would be 16777218.0 |}]

(* ---- adversarial values ---------------------------------------------------- *)

let%expect_test "Kernel_eval: signed zero, infinities, NaN and subnormals" =
  (* Bitwise agreement is the whole claim here: [Float.equal] would call every
     one of these pairs equal and the comparison would prove nothing. *)
  let vals =
    [| 0.0; -0.0; infinity; neg_infinity; Float.nan; 5e-324; 1.4e-45; -1.0 |]
  in
  let g =
    build "adversarial"
      (fun r -> [ r ])
      Graph_builder.(
        let* x = input ~shape:(s1c 8) () in
        let* y = input ~shape:(s1c 8) () in
        add x y)
  in
  let t = fill (s1c 8) (fun c -> vals.(Dim.to_int (Vec6.get c Axis.C))) in
  let inputs = List.map (fun id -> (id, t)) g.Graph.inputs in
  Format.printf "@[<v>%a@]@." (pp_result pp_agreements) (compare_all ~inputs g);
  [%expect {|
    value_at: true
    ground: true
    direct: true |}]

let%expect_test "Kernel_eval: a Filled constant is stored through f32" =
  (* [Stage_program.ground] materialises a const rather than handing the OCaml
     float through, so the evaluator must too. Pinned via the round trip: 2^24+1
     is the smallest positive integer f32 cannot hold. *)
  let sg =
    Tensor_sig.create ~id:(Tensor_id.of_int 1) ~name:"" ~shape:(s1c 1)
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let k =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create
         ~inputs:
           [
             {
               Kernel.Input.id = Tensor_id.of_int 1;
               sg;
               binding = Kernel.Binding.Filled 16777217.0;
             };
           ]
         ~values:
           [
             {
               Kernel.Value.id = Tensor_id.of_int 2;
               sg =
                 Tensor_sig.create ~id:(Tensor_id.of_int 2) ~name:""
                   ~shape:(s1c 1) ~fmt:(Payload.Fmt Payload.F32) ();
               computation =
                 Region_program.pixel
                   (Expr.Value.load
                      (Expr_bridge.source_of_id (Tensor_id.of_int 1))
                      (Expr_bridge.coord_of_vec6 Symbolic.out_vec));
               result = Kernel.Result_conversion.Round_f32;
             };
           ]
         ~outputs:[ Tensor_id.of_int 2 ]
         ())
  in
  let out =
    Err.or_raise ~pp_error:Kernel_eval.pp_error
      (Kernel_eval.run k ~bind:(fun _ -> None))
  in
  Format.printf "filled 2^24+1 reads back as %.1f (unrounded would be %.1f)@."
    (Tensor.read_at_raw
       (Tensor_id.Map.find (Tensor_id.of_int 2) out)
       (fun _ -> 0))
    16777217.0;
  [%expect
    {| filled 2^24+1 reads back as 16777216.0 (unrounded would be 16777217.0) |}]

(* ---- the binding boundary --------------------------------------------------- *)

let%expect_test "Kernel_eval: a bound tensor is validated against its signature"
    =
  let sg ?(fmt = Payload.Fmt Payload.F32) ?quant id shape =
    Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape ~fmt ?quant ()
  in
  let kernel ?(input_sg = sg 0 (s1c 3)) () =
    Err.or_raise ~pp_error:Kernel.pp_error
      (Kernel.create
         ~inputs:
           [
             {
               Kernel.Input.id = Tensor_id.of_int 0;
               sg = input_sg;
               binding = Kernel.Binding.Caller;
             };
           ]
         ~values:
           [
             {
               Kernel.Value.id = Tensor_id.of_int 2;
               sg = sg 2 (s1c 3);
               computation =
                 Region_program.pixel
                   (Expr.Value.load
                      (Expr_bridge.source_of_id (Tensor_id.of_int 0))
                      (Expr_bridge.coord_of_vec6 Symbolic.out_vec));
               result = Kernel.Result_conversion.Round_f32;
             };
           ]
         ~outputs:[ Tensor_id.of_int 2 ]
         ())
  in
  let case name ?input_sg bind =
    Format.printf "%s: %a@." name
      (Core.Pretty.err_result
         ~ok:(fun ppf _ -> Fmt.string ppf "ok")
         ~error:Kernel_eval.pp_error)
      (Kernel_eval.run (kernel ?input_sg ()) ~bind)
  in
  case "unbound" (fun _ -> None);
  case "right shape" (fun _ -> Some (ramp (s1c 3)));
  case "wrong shape" (fun _ -> Some (ramp (s1c 4)));
  case "wrong format" (fun _ ->
      Some
        (Tensor.Tensor
           {
             Tensor.shape = s1c 3;
             payload =
               {
                 Payload.fmt = Payload.F16;
                 quant = Payload.No_quant;
                 data = Bigarray.(Array1.create int16_unsigned c_layout 3);
               };
           }));
  (* A public record with a SHORT Bigarray: the shape matches, so every other
     check passes, and [Tensor.read] would index past the end at an otherwise
     in-range offset. *)
  case "short storage" (fun _ ->
      Some
        (Tensor.Tensor
           {
             Tensor.shape = s1c 3;
             payload =
               {
                 Payload.fmt = Payload.F32;
                 quant = Payload.No_quant;
                 data = Bigarray.(Array1.create float32 c_layout 1);
               };
           }));
  [%expect
    {|
    unbound: no binding for input t0
    right shape: ok
    wrong shape: t0: bound tensor has the wrong shape
    wrong format: t0: bound tensor has the wrong format
    short storage: t0: bound tensor holds 1 cells, expected 3 |}]

let%expect_test "Kernel_eval: value_at rejects unknown ids and bad coordinates"
    =
  let g =
    build "seq"
      (fun r -> [ r ])
      Graph_builder.(
        let* x = input ~shape:(s1c 3) () in
        relu x)
  in
  let prog = Eval_symbolic.run g in
  let k =
    Err.or_raise ~pp_error:Kernel_adapt.pp_error
      (Kernel_adapt.of_stage_program prog)
  in
  let bind id =
    if Tensor_id.equal id (List.hd g.Graph.inputs) then Some (ramp (s1c 3))
    else None
  in
  let case name id c =
    Format.printf "%s: %a@." name
      (Core.Pretty.err_result ~ok:(Fmt.fmt "%g") ~error:Kernel_eval.pp_error)
      (Kernel_eval.value_at k ~bind id
         (Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c))
  in
  let v = (List.hd k.Kernel.values).Kernel.Value.id in
  case "in range" v 2;
  (* [Expr.Eval] treats the OUTPUT coordinate as given and checks only what a
     load reads, so this bound is [value_at]'s own obligation. *)
  case "out of range" v 3;
  case "unknown id" (Tensor_id.of_int 99) 0;
  [%expect
    {|
    in range: 3
    out of range: t1[0,0,0,0,0,3] out of range on axis C: 3
    unknown id: t99 names no value |}]
