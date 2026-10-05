open Ssa_ir

type error = [ `Unsupported of Ssa_unsupported.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unsupported u -> Ssa_unsupported.pp fmt u

let extents (sg : Tensor_sig.t) =
  Expr.Coord.of_fn (fun a ->
      Int64.of_int (Dim.to_int (Vec6.get sg.Tensor_sig.shape a)))

let format_of (sg : Tensor_sig.t) =
  let (Payload.Fmt f) = sg.Tensor_sig.fmt in
  match f with
  | Payload.Bool -> Some Ssa_format.Bool
  | Payload.F32 -> Some Ssa_format.F32
  | Payload.I64 -> Some Ssa_format.I64
  | _ -> None

let format_name (sg : Tensor_sig.t) =
  let (Payload.Fmt f) = sg.Tensor_sig.fmt in
  Payload.fmt_name f

let buffer (sg : Tensor_sig.t) format role =
  {
    Ssa_buffer.id = Ssa_lower_ctx.buffer_id sg.Tensor_sig.id;
    extents = extents sg;
    format;
    role;
  }

(* The inputs a caller binds, in input order, each as the buffer it declares or
   the reason it has none. A [Filled] input is a constant: it has no buffer, and
   a read of it is refused until its bounds check is covered. *)
let input_sources (k : Kernel.t) =
  List.fold_left
    (fun (buffers, sources) (i : Kernel.Input.t) ->
      let sg = i.Kernel.Input.sg in
      let id = sg.Tensor_sig.id in
      match i.Kernel.Input.binding with
      | Kernel.Binding.Caller | Kernel.Binding.Captured_constant -> (
          match format_of sg with
          | Some format ->
              let b = buffer sg format Ssa_buffer.Input in
              ( b :: buffers,
                Tensor_id.Map.add id (Ssa_lower_ctx.Buffer b) sources )
          | None ->
              ( buffers,
                Tensor_id.Map.add id
                  (Ssa_lower_ctx.Unsupported_format (format_name sg))
                  sources ))
      | Kernel.Binding.Filled _ | Kernel.Binding.Filled_i64 _ ->
          (buffers, Tensor_id.Map.add id Ssa_lower_ctx.Filled sources))
    ([], Tensor_id.Map.empty) k.Kernel.inputs

let encode_of : Kernel.Result_conversion.t -> Ssa_op.Encode.t = function
  | Kernel.Result_conversion.Nonzero_bool -> Ssa_op.Encode.Bool_nonzero
  | Kernel.Result_conversion.Round_f32 -> Ssa_op.Encode.F32_round

(* The consumer's body, with its one virtual producer (if any) inlined by the
   capture-safe elaborator. Lowering never substitutes a producer body itself:
   the proof layer compares that same tree, and [Kernel_eval], which reaches the
   producer by recursion instead, stays the independent oracle for this path. *)
let body ctx (plan : Fusion_plan.t) (v : Kernel.Value.t) pixel =
  let uses =
    Kernel.Use.Set.elements
      (Kernel.Use.Set.filter
         (fun u -> Tensor_id.equal u.Kernel.Use.consumer v.Kernel.Value.id)
         plan.Fusion_plan.virtual_uses)
  in
  match uses with
  | [] -> pixel
  | [ use ] -> (
      match Kernel_elab.elaborate plan.Fusion_plan.kernel use with
      | Ok elaborated -> elaborated
      | Error _ -> Ssa_lower_ctx.refuse ctx Ssa_unsupported.Virtual_use)
  | _ :: _ :: _ -> Ssa_lower_ctx.refuse ctx Ssa_unsupported.Virtual_use

(* The dense six-axis nest, then the stored value at the cell. *)
let nest (ctx : Ssa_lower_ctx.t) plan (v : Kernel.Value.t) pixel =
  let sg = v.Kernel.Value.sg in
  let out_id = Ssa_lower_ctx.buffer_id v.Kernel.Value.id in
  let rec go (ctx : Ssa_lower_ctx.t) inductions = function
    | [] ->
        let axes =
          Expr.Coord.of_fn (fun a -> List.assoc a (List.rev inductions))
        in
        let ctx = { ctx with Ssa_lower_ctx.axes = Some axes } in
        let x =
          Ssa_lower_value.value ctx
            (Kernel.Result_conversion.apply v.Kernel.Value.result
               (body ctx plan v pixel))
        in
        Ssa_builder.store_f64 ctx.Ssa_lower_ctx.b out_id
          ~encode:(encode_of v.Kernel.Value.result)
          (Ssa_builder.Coord axes) x
    | a :: rest ->
        let lo = Ssa_builder.index ctx.Ssa_lower_ctx.b 0L in
        let hi =
          Ssa_builder.index ctx.Ssa_lower_ctx.b
            (Int64.of_int (Dim.to_int (Vec6.get sg.Tensor_sig.shape a)))
        in
        let Ssa_builder.Nil =
          Ssa_builder.for_ ctx.Ssa_lower_ctx.b ~lo ~hi ~init:Ssa_builder.Nil
            (fun b i Ssa_builder.Nil ->
              go { ctx with Ssa_lower_ctx.b } ((a, i) :: inductions) rest;
              Ssa_builder.Nil)
        in
        ()
  in
  go { ctx with Ssa_lower_ctx.at = v.Kernel.Value.id } [] Expr.Axis.all

let lower (plan : Fusion_plan.t) =
  Err.Escape.with_escape @@ fun esc ->
  let k = plan.Fusion_plan.kernel in
  let stored (v : Kernel.Value.t) =
    Tensor_id.Set.mem v.Kernel.Value.id plan.Fusion_plan.stores
  in
  let inputs, sources = input_sources k in
  let refuse_at at construct =
    Err.Escape.throw esc
      (`Unsupported { Ssa_unsupported.at; construct } : error)
  in
  (match k.Kernel.values_i64 with
  | v :: _ -> refuse_at v.Kernel.Value_i64.id Ssa_unsupported.Int64_value
  | [] -> ());
  let outputs =
    List.filter_map
      (fun (v : Kernel.Value.t) ->
        if not (stored v) then None
        else
          match format_of v.Kernel.Value.sg with
          | Some format ->
              Some (buffer v.Kernel.Value.sg format Ssa_buffer.Output)
          | None -> invalid_arg "Ssa_lower: output buffer format")
      k.Kernel.values
  in
  let sources =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        Tensor_id.Map.add
          (Tensor_id.of_int (b.Ssa_buffer.id :> int))
          (Ssa_lower_ctx.Buffer b) m)
      sources outputs
  in
  let runs =
    Region_group.runs
      ~computation:(fun (v : Kernel.Value.t) -> v.Kernel.Value.computation)
      k.Kernel.values
  in
  let buffers = List.rev inputs @ outputs in
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (Ssa_builder.program ~buffers (fun b ->
         let ctx =
           {
             Ssa_lower_ctx.esc;
             at = Tensor_id.of_int 0;
             b;
             sources;
             axes = None;
             reducers = Expr.Reduce_var.Map.empty;
           }
         in
         List.iter
           (function
             | Region_group.Run.Solo v when stored v -> (
                 match
                   Region_group.Ref.pixel_expression v.Kernel.Value.computation
                 with
                 | Some pixel -> nest ctx plan v pixel
                 | None ->
                     refuse_at v.Kernel.Value.id Ssa_unsupported.Region_program)
             | Region_group.Run.Solo _ -> ()
             | Region_group.Run.Group (_, members) -> (
                 match List.filter (fun (_, v) -> stored v) members with
                 | [] -> ()
                 | (_, first) :: _ ->
                     refuse_at first.Kernel.Value.id
                       Ssa_unsupported.Region_program))
           runs))
