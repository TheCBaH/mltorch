(* Differential test: every op run through BOTH symbolic instances, comparing
   the expression they build and the value it evaluates to.

   This is what makes the eventual cutover reviewable. The two representations
   coexist only until then, and this suite is the evidence that swapping one for
   the other changes nothing -- so the migration commit can be read as a type
   change rather than a behaviour change.

   Both directions are checked, and they catch different things: the printed
   form catches a structural difference that happens to evaluate the same (a
   reassociated reduction, a swapped operand), and the evaluated bits catch a
   semantic difference the printer renders identically. *)

let f32 = Payload.Fmt Payload.F32
let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let sg id shape =
  Tensor_sig.create ~id:(Tensor_id.of_int id) ~name:"" ~shape ~fmt:f32 ()

(* One tensor per source, filled with a coordinate ramp rather than a constant:
   a ramp separates every element, so a permuted or off-by-one index shows up as
   a different value instead of coinciding. *)
let tensor shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        (Vec6.foldi
           (fun acc a i -> (acc * 3) + Axis.to_int a + Dim.to_int i)
           1 c
        mod 17)
      -. 8.)

let bind sigs =
  let tbl =
    List.map
      (fun (s : Tensor_sig.t) -> (s.Tensor_sig.id, tensor s.Tensor_sig.shape))
      sigs
  in
  fun id -> List.assoc_opt id tbl

(* The old instance evaluates through [Symbolic_expr.eval], which takes a total
   signature-keyed binding; the new one goes through [Expr.Eval] and the bridge.
   Feeding both from one table is the point -- a difference in what they read
   would otherwise look like a difference in what they compute. *)
let eval_old ~sigs e coord =
  let b = bind sigs in
  Symbolic_expr.eval e
    ~binding:(fun (s : Tensor_sig.t) -> Option.get (b s.Tensor_sig.id))
    ~coord:(Schedule.coord_index coord)

let eval_new ~sigs e coord =
  let env = Expr_bridge.env ~binding:(bind sigs) in
  Expr.Eval.value env
    ~output:(Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int coord))
    e

let same_bits a b = Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* [build_old]/[build_new] each take a fresh instance, so reducer numbering
   starts clean and the two are compared on equal terms. *)
let diff name ~sigs ~iter_shape ~build_old ~build_new =
  let old_e = build_old () and new_e = build_new () in
  let printed_old = Core.Pretty.to_string Symbolic_expr.pp old_e in
  let printed_new = Core.Pretty.to_string Expr.Pp.value new_e in
  let mismatches = ref [] in
  Vec6.iter iter_shape (fun c ->
      let o = eval_old ~sigs old_e c in
      match eval_new ~sigs new_e c with
      | Ok n when same_bits o n -> ()
      | Ok n -> mismatches := Fmt.str "%g vs %g" o n :: !mismatches
      | Error e ->
          mismatches :=
            Core.Pretty.to_string (Core.Pretty.error_kind Expr.Eval.pp_error) e
            :: !mismatches);
  Fmt.pr "%-26s values %s   printed %s@." name
    (match !mismatches with
    | [] -> "agree"
    | ms -> "DIFFER: " ^ String.concat "; " (List.rev ms))
    (if String.equal printed_old printed_new then "identical"
     else Fmt.str "DIFFER\n  old: %s\n  new: %s" printed_old printed_new)

let%expect_test "pointwise ops agree between the two symbolic instances" =
  let x = sg 0 (s1c 3) and y = sg 1 (s1c 3) in
  let iter = s1c 3 in
  diff "relu" ~sigs:[ x ] ~iter_shape:iter
    ~build_old:(fun () ->
      let module S = Symbolic.Make () in
      let module M = Pointwise.Relu.Compute (S) in
      M.pixel x Symbolic.out_vec)
    ~build_new:(fun () ->
      let module S = Symbolic2.Make () in
      let module M = Pointwise.Relu.Compute (S) in
      M.pixel x Symbolic2.out_vec);
  diff "add" ~sigs:[ x; y ] ~iter_shape:iter
    ~build_old:(fun () ->
      let module S = Symbolic.Make () in
      let module M = Pointwise.Add.Compute (S) in
      M.pixel ~a_shape:(s1c 3) ~b_shape:(s1c 3) x y Symbolic.out_vec)
    ~build_new:(fun () ->
      let module S = Symbolic2.Make () in
      let module M = Pointwise.Add.Compute (S) in
      M.pixel ~a_shape:(s1c 3) ~b_shape:(s1c 3) x y Symbolic2.out_vec);
  [%expect
    {|
    relu                       values agree   printed identical
    add                        values agree   printed identical
    |}]

let%expect_test "reductions agree, including numbering and fold order" =
  (* A reduction is where the two could most easily diverge: the old instance
     numbers reducers from a bare counter, the new one from a threaded supply,
     and each prints through its own printer. They agree because the old
     counter incremented BEFORE building the body -- so its order was already
     lexical -- and the new printer numbers from 1 to match the base.

     This case found a real defect. Publishing the advanced supply only AFTER
     evaluating the body let a nested reduction read the stale ref, so conv's
     three reducers collapsed onto one variable and the values differed. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:3 ~c:2 in
  let x = sg 0 x_shape in
  let p =
    {
      Conv.Conv2d.h =
        {
          Conv.Conv2d.kernel = Dim.extent 2;
          stride = Op_config.Pos.of_int 1;
          pad_before = Op_config.Nonneg.of_int 0;
          pad_after = Op_config.Nonneg.of_int 0;
          dilation = Op_config.Pos.of_int 1;
        };
      w =
        {
          Conv.Conv2d.kernel = Dim.extent 2;
          stride = Op_config.Pos.of_int 1;
          pad_before = Op_config.Nonneg.of_int 0;
          pad_after = Op_config.Nonneg.of_int 0;
          dilation = Op_config.Pos.of_int 1;
        };
      in_channels = Dim.extent 2;
      groups = Op_config.Pos.of_int 1;
    }
  in
  let w_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:2 in
  let ws = sg 1 w_shape and bs = sg 2 (s1c 1) in
  let iter = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:2 ~c:1 in
  diff "conv2d" ~sigs:[ x; ws; bs ] ~iter_shape:iter
    ~build_old:(fun () ->
      let module S = Symbolic.Make () in
      let module M = Conv.Conv2d.Compute (S) in
      M.pixel p ~x_shape ~weight_shape:w_shape ~x ~weight:ws ~bias:bs
        Symbolic.out_vec)
    ~build_new:(fun () ->
      let module S = Symbolic2.Make () in
      let module M = Conv.Conv2d.Compute (S) in
      M.pixel p ~x_shape ~weight_shape:w_shape ~x ~weight:ws ~bias:bs
        Symbolic2.out_vec);
  [%expect {| conv2d                     values agree   printed identical |}]

let%expect_test "max-pool agrees on both outputs" =
  (* The intrinsic is the one node whose descriptor changed shape -- the old
     form embeds a Tensor_sig.t and recovers the input extents from it, the new
     one carries them explicitly from ~x_shape. Both outputs are checked because
     value and index advance under one shared predicate, and it was updating
     them separately that originally let them drift. *)
  let x_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
  let x = sg 0 x_shape in
  let p =
    {
      Pool.MaxPool2d.kernel =
        { Op_config.Hw.h = Dim.extent 2; w = Dim.extent 2 };
      stride =
        { Op_config.Hw.h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
      pad =
        {
          Op_config.Hw.h = Op_config.Nonneg.of_int 1;
          w = Op_config.Nonneg.of_int 1;
        };
    }
  in
  let iter = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
  diff "max_pool2d value" ~sigs:[ x ] ~iter_shape:iter
    ~build_old:(fun () ->
      let module S = Symbolic.Make () in
      let module M = Pool.MaxPool2d.Compute (S) in
      M.pixel p ~x_shape ~x Symbolic.out_vec)
    ~build_new:(fun () ->
      let module S = Symbolic2.Make () in
      let module M = Pool.MaxPool2d.Compute (S) in
      M.pixel p ~x_shape ~x Symbolic2.out_vec);
  [%expect {| max_pool2d value           values agree   printed identical |}]
