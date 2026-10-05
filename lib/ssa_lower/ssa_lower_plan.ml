open Ssa_ir

type error = [ `Unsupported of Ssa_unsupported.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unsupported u -> Ssa_unsupported.pp fmt u

let buffer sg role =
  match Ssa_sig.buffer sg role with
  | Some b -> b
  | None -> invalid_arg "Ssa_lower: a buffer whose format has no declaration"

(* The inputs a caller binds, in input order, each as the buffer it declares or
   the reason it has none. A [Filled] input is a constant with no data, but a
   read of it still checks its coordinate against its shape, so it is declared
   as a scratch buffer that is never written: the shape is all it is for. The
   declared buffers are the caller-bound inputs and then the fills. *)
let input_sources (k : Kernel.t) =
  let fill sg id make (buffers, sources) =
    match Ssa_sig.buffer sg Ssa_buffer.Scratch with
    | Some b -> (buffers @ [ b ], Tensor_id.Map.add id (make b) sources)
    | None ->
        ( buffers,
          Tensor_id.Map.add id
            (Ssa_lower_ctx.Unsupported_format (Ssa_sig.format_name sg))
            sources )
  in
  let bound, fills =
    List.partition
      (fun (i : Kernel.Input.t) ->
        match i.Kernel.Input.binding with
        | Kernel.Binding.Caller | Kernel.Binding.Captured_constant -> true
        | Kernel.Binding.Filled _ | Kernel.Binding.Filled_i64 _ -> false)
      k.Kernel.inputs
  in
  let bound_buffers, bound_sources =
    List.fold_left
      (fun (buffers, sources) (i : Kernel.Input.t) ->
        let sg = i.Kernel.Input.sg in
        let id = sg.Tensor_sig.id in
        match i.Kernel.Input.binding with
        | Kernel.Binding.Caller | Kernel.Binding.Captured_constant -> (
            match Ssa_sig.buffer sg Ssa_buffer.Input with
            | Some b ->
                ( b :: buffers,
                  Tensor_id.Map.add id (Ssa_lower_ctx.Buffer b) sources )
            | None ->
                ( buffers,
                  Tensor_id.Map.add id
                    (Ssa_lower_ctx.Unsupported_format (Ssa_sig.format_name sg))
                    sources ))
        | Kernel.Binding.Filled _ | Kernel.Binding.Filled_i64 _ ->
            (buffers, sources))
      ([], Tensor_id.Map.empty) bound
  in
  List.fold_left
    (fun acc (i : Kernel.Input.t) ->
      let sg = i.Kernel.Input.sg in
      let id = sg.Tensor_sig.id in
      match i.Kernel.Input.binding with
      | Kernel.Binding.Filled v ->
          fill sg id (fun b -> Ssa_lower_ctx.Fill (b, v)) acc
      | Kernel.Binding.Filled_i64 v ->
          fill sg id (fun b -> Ssa_lower_ctx.Fill_i64 (b, v)) acc
      | Kernel.Binding.Caller | Kernel.Binding.Captured_constant -> acc)
    (List.rev bound_buffers, bound_sources)
    fills

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

(* The dense six-axis nest, N outermost and C innermost, as [Vec6.iter] visits a
   tensor. [store] lowers and writes the cell's value, inside the nest. A cell
   that reads the scan meter starts with a fresh one, which only lowering the
   cell can tell: the cell is built once into a block that is thrown away to
   find out. *)
let nest_with (ctx : Ssa_lower_ctx.t) ~id ~(sg : Tensor_sig.t) store =
  let uses_meter =
    let probed = ref false in
    Ssa_builder.probe ctx.Ssa_lower_ctx.b (fun b ->
        let zero = Ssa_builder.index b 0L in
        let axes = Expr.Coord.of_fn (fun _ -> zero) in
        let meter = ref false in
        store
          { ctx with Ssa_lower_ctx.b; axes = Some axes; at = id; meter }
          axes;
        probed := !meter);
    !probed
  in
  let rec go (ctx : Ssa_lower_ctx.t) inductions = function
    | [] ->
        let axes =
          Expr.Coord.of_fn (fun a -> List.assoc a (List.rev inductions))
        in
        if uses_meter then Ssa_builder.meter_reset ctx.Ssa_lower_ctx.b;
        store { ctx with Ssa_lower_ctx.axes = Some axes } axes
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
  go { ctx with Ssa_lower_ctx.at = id } [] Expr.Axis.all

let nest ctx plan (v : Kernel.Value.t) pixel =
  nest_with ctx ~id:v.Kernel.Value.id ~sg:v.Kernel.Value.sg (fun ctx axes ->
      let x =
        Ssa_lower_value.value ctx
          (Kernel.Result_conversion.apply v.Kernel.Value.result
             (body ctx plan v pixel))
      in
      Ssa_builder.store_f64 ctx.Ssa_lower_ctx.b
        (Ssa_lower_ctx.buffer_id v.Kernel.Value.id)
        ~encode:(encode_of v.Kernel.Value.result)
        (Ssa_builder.Coord axes) x)

(* An int64 value is exact end to end: no conversion, stored as int64. *)
let nest_i64 ctx (v : Kernel.Value_i64.t) =
  nest_with ctx ~id:v.Kernel.Value_i64.id ~sg:v.Kernel.Value_i64.sg
    (fun ctx axes ->
      let x = Ssa_lower_value.value_i64 ctx v.Kernel.Value_i64.pixel in
      Ssa_builder.store_i64 ctx.Ssa_lower_ctx.b
        (Ssa_lower_ctx.buffer_id v.Kernel.Value_i64.id)
        (Ssa_builder.Coord axes) x)

(* The reference rejects a Region program over its admission budget (local
   slots, scan state, updates per key) before it evaluates anything, by
   [Region_program.preflight] on the converted program. Refusing what it rejects
   keeps the two from disagreeing about a program neither is meant to run. *)
let region_unit ~refuse_at (v : Kernel.Value.t) program
    ~(limits : Kernel.Limits.t) =
  match
    Region_execution.lower_region ~max_size:limits.Kernel.Limits.max_size
      ~max_depth:limits.Kernel.Limits.max_depth
      ~max_local_slots:limits.Kernel.Limits.max_local_slots
      ~scan_limits:(Kernel.Limits.scan_limits limits)
      ~output_shape:v.Kernel.Value.sg.Tensor_sig.shape
      (Region_program.with_output program
         (Kernel.Result_conversion.apply v.Kernel.Value.result
            (Region_program.output program)))
  with
  | Ok _ -> ()
  | Error _ -> refuse_at v.Kernel.Value.id Ssa_unsupported.Region_admission

(* A run of grouped values: one shared recurrence per canonical key, one store
   per SELECTED member. The reference re-validates each member's converted
   emitter before running, and so does this. *)
let group_unit ~refuse_at (first : Kernel.Value.t) g selected
    ~(limits : Kernel.Limits.t) =
  let converted =
    Region_group.map_outputs g (fun ordinal output ->
        match List.assoc_opt ordinal selected with
        | Some (v : Kernel.Value.t) ->
            Kernel.Result_conversion.apply v.Kernel.Value.result output
        | None -> output)
  in
  match
    Region_execution.lower_group ~max_size:limits.Kernel.Limits.max_size
      ~max_depth:limits.Kernel.Limits.max_depth
      ~max_local_slots:limits.Kernel.Limits.max_local_slots
      ~scan_limits:(Kernel.Limits.scan_limits limits)
      converted
  with
  | Ok _ -> ()
  | Error _ -> refuse_at first.Kernel.Value.id Ssa_unsupported.Region_admission

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
  let outputs =
    List.filter_map
      (fun (v : Kernel.Value.t) ->
        if stored v then Some (buffer v.Kernel.Value.sg Ssa_buffer.Output)
        else None)
      k.Kernel.values
  in
  let i64_outputs =
    List.map
      (fun (v : Kernel.Value_i64.t) ->
        buffer v.Kernel.Value_i64.sg Ssa_buffer.Output)
      k.Kernel.values_i64
  in
  let sources =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        Tensor_id.Map.add
          (Tensor_id.of_int (b.Ssa_buffer.id :> int))
          (Ssa_lower_ctx.Buffer b) m)
      sources (outputs @ i64_outputs)
  in
  let runs =
    Region_group.runs
      ~computation:(fun (v : Kernel.Value.t) -> v.Kernel.Value.computation)
      k.Kernel.values
  in
  let buffers = inputs @ outputs @ i64_outputs in
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (Ssa_builder.program
       ~scan_limits:(Kernel.Limits.scan_limits k.Kernel.limits) ~buffers
       (fun b ->
         let ctx =
           {
             Ssa_lower_ctx.esc;
             at = Tensor_id.of_int 0;
             b;
             sources;
             axes = None;
             reducers = Expr.Reduce_var.Map.empty;
             locals = Expr.Local_var.Map.empty;
             meter = ref false;
           }
         in
         (* An int64 value runs before the first float value that reads it, and
            after everything it reads: [Kernel.create] checked the two lists as
            one dependency graph, so an entry reads only earlier floats and
            earlier entries. The rest follow the floats, in list order.
            [Kernel_eval] instead produces one on demand, when the first load of
            it is evaluated; both fail with the same row, but a float value that
            would fail before reaching such a load fails first there and after
            it here. *)
         let emitted = ref Tensor_id.Set.empty in
         let rec emit_i64 (v : Kernel.Value_i64.t) =
           if not (Tensor_id.Set.mem v.Kernel.Value_i64.id !emitted) then (
             emitted := Tensor_id.Set.add v.Kernel.Value_i64.id !emitted;
             Expr.Source.Set.iter
               (fun src ->
                 match
                   Tensor_id.Map.find_opt
                     (Expr_bridge.id_of_source src)
                     k.Kernel.by_id_i64
                 with
                 | Some dep -> emit_i64 dep
                 | None -> ())
               (Expr.Fold.sources_i64 v.Kernel.Value_i64.pixel);
             nest_i64 ctx v)
         in
         let reads_i64 sources =
           List.iter
             (fun (v : Kernel.Value_i64.t) ->
               if
                 Expr.Source.Set.mem
                   (Expr_bridge.source_of_id v.Kernel.Value_i64.id)
                   sources
               then emit_i64 v)
             k.Kernel.values_i64
         in
         List.iter
           (function
             | Region_group.Run.Solo v when stored v -> (
                 reads_i64 (Region_group.Ref.sources v.Kernel.Value.computation);
                 match
                   Region_group.Ref.pixel_expression v.Kernel.Value.computation
                 with
                 | Some pixel -> nest ctx plan v pixel
                 | None -> (
                     let limits = k.Kernel.limits in
                     match
                       Region_group.Ref.project
                         ~max_size:limits.Kernel.Limits.max_size
                         ~max_depth:limits.Kernel.Limits.max_depth
                         v.Kernel.Value.computation
                     with
                     | Ok program ->
                         region_unit ~refuse_at v program ~limits;
                         Ssa_lower_region.lower ctx v program
                     | Error _ ->
                         refuse_at v.Kernel.Value.id
                           Ssa_unsupported.Region_program))
             | Region_group.Run.Solo _ -> ()
             | Region_group.Run.Group (g, members) -> (
                 match List.filter (fun (_, v) -> stored v) members with
                 | [] -> ()
                 | (_, first) :: _ as selected ->
                     reads_i64
                       (List.fold_left
                          (fun acc (_, (v : Kernel.Value.t)) ->
                            Expr.Source.Set.union acc
                              (Region_group.Ref.sources
                                 v.Kernel.Value.computation))
                          Expr.Source.Set.empty selected);
                     group_unit ~refuse_at first g selected
                       ~limits:k.Kernel.limits;
                     Ssa_lower_region.lower_group
                       { ctx with Ssa_lower_ctx.at = first.Kernel.Value.id }
                       g selected))
           runs;
         List.iter emit_i64 k.Kernel.values_i64))
