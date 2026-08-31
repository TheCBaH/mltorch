let broadcast_coord ~input ~output coord =
  Vec6.of_fn (fun axis ->
      if
        Dim.equal (Vec6.get input.Tensor_sig.shape axis) Dim.one
        && not (Dim.equal (Vec6.get output.Tensor_sig.shape axis) Dim.one)
      then Dim.index 0
      else Vec6.get coord axis)

(* A leaf can be requested through an operation whose result broadcasts it.
   [ground]'s Apply arms normally project that coordinate before recurring, but
   a literal is also a valid exported plan value in its own right. Clamp its
   singleton axes here as the final safety boundary: [Tensor.read] is strict,
   and reading epsilon [C=1] at a consumer's [C=17] is neither symbolic nor
   valid concrete evaluation. *)
let leaf_coord output coord =
  Vec6.of_fn (fun axis ->
      if Dim.equal (Vec6.get output.Tensor_sig.shape axis) Dim.one then
        Dim.index 0
      else Vec6.get coord axis)

(* The inverse of [Vec6.offset]: row-major, C innermost, matching
   [Reshape.Compute.delinearize]'s convention -- but over a concrete [coord]
   rather than a [Semantics.SEMANTICS] index, since grounding only traces a
   coordinate back to its source and never loads a value. *)
let reshape_source_coord ~(input : Tensor_sig.t) ~(output : Tensor_sig.t) coord
    =
  let off = (Vec6.offset output.Tensor_sig.shape coord :> int) in
  let strides, _ =
    List.fold_left
      (fun (acc, prod) axis ->
        let extent = (Vec6.get input.Tensor_sig.shape axis :> int) in
        ((axis, prod) :: acc, prod * extent))
      ([], 1) (List.rev Axis.all)
  in
  Vec6.of_fn (fun axis ->
      let stride = List.assoc axis strides in
      let extent = (Vec6.get input.Tensor_sig.shape axis :> int) in
      Dim.index (off / stride mod extent))

(* Mirrors [Pointwise_binary.Pow.Compute.pixel]'s six ATen-special-cased
   exponents plus its [exp(scalar * log x)] fallback, exactly -- the
   grounding must stay the same expression PowKernel.cpp special-cases, not a
   generic [pow] node, or map verification would compare a different
   function than the one materialization (which calls this same Compute
   functor via Eval_direct) actually produces. *)
let pow_expr ~scalar (v : Ground_expr.t) : Ground_expr.t =
  let mul a b = Ground_expr.Binary (Expr.Value.Mul, a, b) in
  let reciprocal v =
    Ground_expr.Binary (Expr.Value.Div, Ground_expr.Const 1., v)
  in
  if scalar = 2.0 then mul v v
  else if scalar = 3.0 then mul (mul v v) v
  else if scalar = -2.0 then reciprocal (mul v v)
  else if scalar = 0.5 then Ground_expr.Unary (Expr.Value.Sqrt, v)
  else if scalar = -0.5 then reciprocal (Ground_expr.Unary (Expr.Value.Sqrt, v))
  else if scalar = -1.0 then reciprocal v
  else
    Ground_expr.Unary
      ( Expr.Value.Exp,
        mul (Ground_expr.Const scalar) (Ground_expr.Unary (Expr.Value.Log, v))
      )

let rec ground store id coord =
  let value =
    Option.value
      (Constant_store.binding store id)
      ~default:(Const_ssa.Value_id.of_tensor_id id)
  in
  match Const_ssa.find (Constant_store.plan store) value with
  | Some (Const_ssa.Leaf { leaf = Const_ssa.Captured capture; _ }) ->
      Some
        (Ground_expr.Cell
           {
             Ground_expr.Cell.origin = Ground_expr.Origin.Capture capture;
             coord;
           })
  | Some
      (Const_ssa.Apply { op = Graph_ir.Permute { Permute.Permute.perm; x }; _ })
    ->
      let input_coord =
        Vec6.of_fn (fun input_axis ->
            Vec6.get coord
              (List.assoc input_axis
                 (List.map
                    (fun (out_axis, in_axis) -> (in_axis, out_axis))
                    perm)))
      in
      ground store x input_coord
  | Some
      (Const_ssa.Apply
         { op = Graph_ir.Reshape { Reshape.Reshape.x; _ }; output }) ->
      Option.bind
        (Const_ssa.sig_of
           (Constant_store.plan store)
           (Const_ssa.Value_id.of_tensor_id x))
        (fun input ->
          ground store x (reshape_source_coord ~input ~output coord))
  | Some
      (Const_ssa.Apply
         { op = Graph_ir.Expand { Pointwise.Expand.x; _ }; output }) ->
      Option.bind
        (Const_ssa.sig_of
           (Constant_store.plan store)
           (Const_ssa.Value_id.of_tensor_id x))
        (fun input -> ground store x (broadcast_coord ~input ~output coord))
  | Some
      (Const_ssa.Apply
         {
           op =
             ( Graph_ir.Add { Pointwise.Bin.a; b }
             | Graph_ir.Sub { Pointwise.Bin.a; b }
             | Graph_ir.Mul { Pointwise.Bin.a; b }
             | Graph_ir.Div { Pointwise.Bin.a; b } ) as op;
           output;
         }) ->
      let operand id =
        Option.bind
          (Const_ssa.sig_of
             (Constant_store.plan store)
             (Const_ssa.Value_id.of_tensor_id id))
          (fun input -> ground store id (broadcast_coord ~input ~output coord))
      in
      Option.bind (operand a) (fun a ->
          Option.map
            (fun b ->
              let op =
                match op with
                | Graph_ir.Add _ -> Expr.Value.Add
                | Graph_ir.Sub _ -> Expr.Value.Sub
                | Graph_ir.Mul _ -> Expr.Value.Mul
                | Graph_ir.Div _ -> Expr.Value.Div
                | _ -> assert false
              in
              Ground_expr.Round (Ground_expr.Binary (op, a, b)))
            (operand b))
  | Some (Const_ssa.Apply { op = Graph_ir.Sqrt { Pointwise.Sqrt.x }; output })
    ->
      Option.bind
        (Const_ssa.sig_of
           (Constant_store.plan store)
           (Const_ssa.Value_id.of_tensor_id x))
        (fun input ->
          Option.map
            (fun x ->
              Ground_expr.Round (Ground_expr.Unary (Expr.Value.Sqrt, x)))
            (ground store x (broadcast_coord ~input ~output coord)))
  | Some
      (Const_ssa.Apply
         {
           op = Graph_ir.Mul_scalar { Pointwise_binary.Scalar_bin.x; scalar };
           _;
         }) ->
      Option.map
        (fun x ->
          Ground_expr.Round
            (Ground_expr.Binary (Expr.Value.Mul, x, Ground_expr.Const scalar)))
        (ground store x coord)
  | Some
      (Const_ssa.Apply
         {
           op = Graph_ir.Add_scalar { Pointwise_binary.Scalar_bin.x; scalar };
           _;
         }) ->
      Option.map
        (fun x ->
          Ground_expr.Round
            (Ground_expr.Binary (Expr.Value.Add, x, Ground_expr.Const scalar)))
        (ground store x coord)
  | Some
      (Const_ssa.Apply
         { op = Graph_ir.Pow { Pointwise_binary.Scalar_bin.x; scalar }; _ }) ->
      Option.map
        (fun x -> Ground_expr.Round (pow_expr ~scalar x))
        (ground store x coord)
  | Some
      (Const_ssa.Leaf
         { leaf = Literal payload | Opaque_materialized payload; output; _ }) ->
      Some
        (Ground_expr.Const
           (Tensor.read_at_raw payload (fun axis ->
                Dim.to_int (Vec6.get (leaf_coord output coord) axis))))
  | Some (Const_ssa.Apply _) | None -> None

let same_fmt (Payload.Fmt a) (Payload.Fmt b) =
  String.equal (Payload.fmt_name a) (Payload.fmt_name b)

let captured_fmt store capture =
  let compatible current fmt =
    match current with
    | None -> Some fmt
    | Some previous when same_fmt previous fmt -> Some previous
    | Some _ -> None
  in
  let seen, result =
    List.fold_left
      (fun (seen, result) (_, definition) ->
        match definition with
        | Const_ssa.Leaf { leaf = Const_ssa.Captured candidate; output; _ }
          when Const_ssa.Capture.equal capture candidate ->
            ( true,
              match result with
              | None when seen -> None
              | current -> compatible current output.Tensor_sig.fmt )
        | Const_ssa.Leaf _ | Const_ssa.Apply _ -> (seen, result))
      (false, None)
      (Const_ssa.bindings (Constant_store.plan store))
  in
  if seen then result else None
