(* Characterization of the [Stage_program] boundary the Kernel IR is built on.
   These tests pin CURRENT behaviour — they pass against the engine as it stands
   and describe exactly what [Kernel]/[Kernel_adapt]/[Kernel_eval] must
   reproduce. Nothing here depends on the kernel modules. See
   .ai/native_kernel_dsl_design.md and the Milestone 0 section of the
   implementation plan. *)

open Graph_ir

type error = [ `Build of Graph_builder.error ]

let pp_error ppf : [< error ] -> unit = function
  | `Build e -> Graph_builder.pp_error ppf e

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error

let lift_build (r : ('a, Graph_builder.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Build e) r

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let conv_params ~in_channels : Conv.Conv2d.params =
  {
    h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int 1;
  }

(* Same layout as [Graph_fixtures.weight_shape]: N=out_channels, H/W=kernel,
   C=in_channels. *)
let weight_shape ~out_channels ~in_channels = s out_channels 1 1 2 2 in_channels

(* ---- source classification ------------------------------------------------

   The rule [Kernel.create] will enforce, stated here against the program that
   produces it: every source in a stage body resolves to a graph input, a
   synthetic constant, or an EARLIER stage. "Every source", not "every load" —
   [Expr.Fold.sources] also reports the source inside a max-pool descriptor,
   which carries no [Load] node at all. *)

let classify (p : Stage_program.t) (st : Stage_program.Stage.t) =
  let input_ids = List.map fst p.Stage_program.inputs in
  let const_ids =
    List.map (fun ((sg : Tensor_sig.t), _) -> sg.id) p.Stage_program.consts
  in
  let earlier =
    (* stages strictly before [st] in the topo order *)
    let rec upto acc = function
      | [] -> acc
      | (s : Stage_program.Stage.t) :: _ when Tensor_id.equal s.id st.id -> acc
      | (s : Stage_program.Stage.t) :: tl -> upto (s.id :: acc) tl
    in
    upto [] p.Stage_program.stages
  in
  let mem ids id = List.exists (Tensor_id.equal id) ids in
  Expr.Source.Set.fold
    (fun src (i, c, e, u) ->
      let id = Expr_bridge.id_of_source src in
      if mem input_ids id then (id :: i, c, e, u)
      else if mem const_ids id then (i, id :: c, e, u)
      else if mem earlier id then (i, c, id :: e, u)
      else (i, c, e, id :: u))
    (Stage_program.Stage.sources st)
    ([], [], [], [])

let pp_ids ppf ids =
  match List.sort Tensor_id.compare ids with
  | [] -> Format.pp_print_string ppf "-"
  | ids ->
      Format.pp_print_string ppf
        (String.concat "," (List.map (Format.asprintf "%a" Tensor_id.pp) ids))

let pp_classification ppf (p : Stage_program.t) =
  List.iter
    (fun (st : Stage_program.Stage.t) ->
      let i, c, e, u = classify p st in
      Format.fprintf ppf
        "%a <- inputs %a | consts %a | stages %a | UNRESOLVED %a@," Tensor_id.pp
        st.id pp_ids i pp_ids c pp_ids e pp_ids u)
    p.Stage_program.stages

let pp_program ppf p = Format.fprintf ppf "@[<v>%a@]" pp_classification p

let%expect_test "Stage_program: Pixel computation retains its expression object"
    =
  let body = Expr.Value.add (Expr.Value.const 1.) (Expr.Value.const 2.) in
  let stage : Stage_program.Stage.t =
    {
      id = Tensor_id.of_int 0;
      sg =
        Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"" ~shape:(s1c 1)
          ~fmt:(Payload.Fmt Payload.F32) ();
      computation = Region_program.pixel body;
    }
  in
  let embedded =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Stage_program.Stage.pixel_body ~max_size:32 ~max_depth:32 stage)
  in
  Format.printf "same_object=%b@." (body == embedded);
  [%expect {| same_object=true |}]

let%expect_test
    "Stage_program: every source resolves to an input, const, or earlier stage"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"chain" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) () in
          let* b = input ~shape:(s1c 3) () in
          let* sum = add a b in
          let* r = relu sum in
          mul r a)
    in
    Err.return (Eval_symbolic.run g)
  in
  Format.printf "%a@." (pp_result pp_program) result;
  [%expect
    {|
    t2 <- inputs t0,t1 | consts - | stages - | UNRESOLVED -
    t3 <- inputs - | consts - | stages t2 | UNRESOLVED -
    t4 <- inputs t0 | consts - | stages t3 | UNRESOLVED - |}]

let%expect_test
    "Stage_program: a max-pool stage's only source is inside its intrinsic" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"pool" ~outputs:(fun (v, _) -> [ v ])
          @@
          let* x = input ~shape:(s 1 1 1 4 4 2) () in
          let params : Pool.MaxPool2dWithIndices.params =
            {
              ceil_mode = false;
              kernel = { h = Dim.extent 2; w = Dim.extent 2 };
              stride =
                { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
              pad =
                { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
            }
          in
          max_pool2d_with_indices params x)
    in
    Err.return (Eval_symbolic.run g)
  in
  (* Both stages of the one node appear, each resolving its source to the input.
     The printed bodies carry NO [tN[...]] load: the dependency lives only in
     the max-pool descriptor. That asymmetry is why Kernel construction must
     validate [Fold.sources] rather than a loads-only query — and why a later
     elaborator, which can only substitute real loads, needs the other one. *)
  Format.printf "%a@."
    (pp_result (fun ppf (p : Stage_program.t) ->
         Format.fprintf ppf "@[<v>%a" pp_classification p;
         List.iter
           (fun (st : Stage_program.Stage.t) ->
             Format.fprintf ppf "%a = %a@," Tensor_id.pp st.id Expr.Pp.value
               (Option.get (Region_program.pixel_expression st.computation)))
           p.Stage_program.stages;
         Format.fprintf ppf "@]"))
    result;
  [%expect
    {|
    t1 <- inputs t0 | consts - | stages - | UNRESOLVED -
    t2 <- inputs t0 | consts - | stages - | UNRESOLVED -
    t1 = max_pool2d_value(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C])
    t2 = max_pool2d_index(t0; k=2x2 s=2x2 p=0x0; out=[N,T,D,H,W,C]) |}]

(* ---- synthetic constants --------------------------------------------------

   An omitted optional operand becomes a [consts] entry with an id past every
   graph id, referenced from the body like any other source. The Kernel adapter
   turns these into ordinary boundary inputs with a [Filled] binding, NOT into
   an evaluator special case. *)

let%expect_test
    "Stage_program: an omitted optional operand becomes a synthetic const" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv_no_bias" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 3 3 2) () in
          let* w =
            input ~shape:(weight_shape ~out_channels:1 ~in_channels:2) ()
          in
          conv2d (conv_params ~in_channels:2) ~x ~weight:w ())
    in
    Err.return (Eval_symbolic.run g)
  in
  Format.printf "%a@."
    (pp_result (fun ppf (p : Stage_program.t) ->
         Format.fprintf ppf "@[<v>max graph id: %a@," pp_ids
           (List.map fst p.Stage_program.inputs
           @ List.map
               (fun (st : Stage_program.Stage.t) -> st.id)
               p.Stage_program.stages);
         List.iter
           (fun ((sg : Tensor_sig.t), v) ->
             Format.fprintf ppf "const %a = %g@," Tensor_id.pp sg.id v)
           p.Stage_program.consts;
         Format.fprintf ppf "%a@]" pp_classification p))
    result;
  [%expect
    {|
    max graph id: t0,t1,t2
    const t3 = 0
    t2 <- inputs t0,t1 | consts t3 | stages - | UNRESOLVED - |}]

(* ---- stage rounding is observable -----------------------------------------

   The load-bearing one. [Schedule.ground] materialises each stage into a
   float32 tensor (`Tensor.materialize`, tensor.ml:87-94), so a stage result
   that is not representable in binary32 is ROUNDED before the next stage reads
   it. 2^24+1 is the smallest positive integer f32 cannot hold.

     with the round:    2^24 -> +1 -> round(2^24+1) = 2^24 -> +1 -> 2^24
     without the round: 2^24 -> +1 ->       2^24+1        -> +1 -> 2^24+2

   Integer-valued throughout, so the golden is identical under js_of_ocaml. A
   fused kernel that drops this round produces 16777218 here, which is exactly
   what the Milestone 3 acceptance test re-uses. *)

let%expect_test "Stage_program: an intermediate stage result is rounded to f32"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"round_chain" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 1) () in
          let* a = add_scalar 1.0 x in
          add_scalar 1.0 a)
    in
    let prog = Eval_symbolic.run g in
    let two24 = 16777216.0 in
    let x = Tensor.materialize (s1c 1) (fun _ -> two24) in
    let bind id = List.assoc id (List.combine g.Graph.inputs [ x ]) in
    let grounded = Stage_program.ground prog ~bind in
    let read id =
      match Tensor_id.Map.find_opt id grounded with
      | None -> nan
      | Some t -> Tensor.read_at_raw t (fun _ -> 0)
    in
    Err.return
      (List.map
         (fun (st : Stage_program.Stage.t) -> (st.id, read st.id))
         prog.Stage_program.stages)
  in
  Format.printf "%a@."
    (pp_result (fun ppf rows ->
         Format.fprintf ppf "@[<v>";
         List.iter
           (fun (id, v) -> Format.fprintf ppf "%a = %.1f@," Tensor_id.pp id v)
           rows;
         Format.fprintf ppf "unrounded binary64 would be %.1f@]"
           (16777216.0 +. 1.0 +. 1.0)))
    result;
  [%expect
    {|
    t1 = 16777216.0
    t2 = 16777216.0
    unrounded binary64 would be 16777218.0 |}]

(* The same store semantics apply to a synthetic constant: [Stage_program.ground]
   materialises it (stage_program.ml:37-41) rather than handing the OCaml float
   through. No CURRENT op fill is inexact in f32 — conv2d fills 0., batch_norm
   fills 1. — so this is pinned at the materialiser, which is where the Kernel
   evaluator must also route [Filled]. *)
let%expect_test "Tensor: materialize stores through float32" =
  let read v =
    Tensor.read_at_raw (Tensor.materialize (s1c 1) (fun _ -> v)) (fun _ -> 0)
  in
  Format.printf "0.1  -> %.17g@." (read 0.1);
  Format.printf "2^24+1 -> %.1f@." (read 16777217.0);
  Format.printf "1.0  -> %.1f@." (read 1.0);
  [%expect
    {|
    0.1  -> 0.10000000149011612
    2^24+1 -> 16777216.0
    1.0  -> 1.0 |}]

(* ---- sparse input_kinds ---------------------------------------------------

   [input_kinds] is sparse BY DESIGN and its effective default is [Input]
   (graph_common.ml:92-100). The adapter must read it with that default; a total
   [Map.find] would silently reclassify an unlisted input as something else or
   raise. *)

let%expect_test "Stage_program: input_kinds is sparse, defaulting to Input" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"kinds" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) () in
          let* k = constant ~shape:(s1c 3) () in
          add x k)
    in
    Err.return (Eval_symbolic.run g)
  in
  let report ppf (p : Stage_program.t) =
    List.iter
      (fun (id, _) ->
        Format.fprintf ppf "%a: listed=%b effective=%a@," Tensor_id.pp id
          (Tensor_id.Map.mem id p.Stage_program.input_kinds)
          Input.pp_kind
          (Option.value
             (Tensor_id.Map.find_opt id p.Stage_program.input_kinds)
             ~default:Input.Input))
      p.Stage_program.inputs
  in
  Format.printf "%a@."
    (pp_result (fun ppf (p : Stage_program.t) ->
         Format.fprintf ppf "@[<v>as built:@,%a" report p;
         (* [Stage_program.t] is a public record and the adapter treats it as
            untrusted, so the unlisted case has to be exercised directly:
            [Graph_builder] happens to record every input, which would let a
            total [Map.find] pass the test above. Stripping the map is the only
            way to make this check falsifiable. *)
         Format.fprintf ppf "@,with input_kinds stripped:@,%a@]" report
           { p with Stage_program.input_kinds = Tensor_id.Map.empty }))
    result;
  [%expect
    {|
    as built:
    t0: listed=true effective=input
    t1: listed=true effective=constant

    with input_kinds stripped:
    t0: listed=false effective=input
    t1: listed=false effective=input |}]

(* ---- reducer identity is expression-local ---------------------------------

   Every stage body is run from [Expr.Builder.initial], so two reduction-bearing
   stages MINT THE SAME ORDINALS. [Expr.Pp] names binders lexically and so hides
   this; [Expr.Fold.binders] does not. No kernel pass may assume a graph-global
   reducer namespace, and anything composing two bodies must freshen first. *)

let%expect_test
    "Stage_program: separately built stages reuse reducer identities" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"two_convs" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 4 4 2) () in
          let* w1 =
            input ~shape:(weight_shape ~out_channels:1 ~in_channels:2) ()
          in
          let* w2 =
            input ~shape:(weight_shape ~out_channels:1 ~in_channels:1) ()
          in
          let* c1 = conv2d (conv_params ~in_channels:2) ~x ~weight:w1 () in
          conv2d (conv_params ~in_channels:1) ~x:c1 ~weight:w2 ())
    in
    Err.return (Eval_symbolic.run g)
  in
  Format.printf "%a@."
    (pp_result (fun ppf (p : Stage_program.t) ->
         let binders (st : Stage_program.Stage.t) =
           Region_program.pixel_expression st.computation
           |> Option.value ~default:(Expr.Value.const 0.)
           |> Expr.Fold.binders
         in
         Format.fprintf ppf "@[<v>";
         List.iter
           (fun (st : Stage_program.Stage.t) ->
             Format.fprintf ppf "%a binders: %s@," Tensor_id.pp st.id
               (String.concat ","
                  (List.map
                     (Format.asprintf "%a" Expr.Reduce_var.pp)
                     (binders st))))
           p.Stage_program.stages;
         (match p.Stage_program.stages with
         | [ a; b ] ->
             let same =
               List.exists
                 (fun x ->
                   List.exists (fun y -> Expr.Reduce_var.equal x y) (binders b))
                 (binders a)
             in
             Format.fprintf ppf "identities collide across stages: %b@," same
         | _ -> Format.fprintf ppf "expected exactly two stages@,");
         Format.fprintf ppf "@]"))
    result;
  [%expect
    {|
    t3 binders: #0,#1,#2
    t4 binders: #0,#1,#2
    identities collide across stages: true |}]
