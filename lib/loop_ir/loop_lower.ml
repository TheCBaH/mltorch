type error = [ `Unsupported of Loop_unsupported.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unsupported u -> Loop_unsupported.pp fmt u

let buffer (sg : Tensor_sig.t) role =
  { Loop_buffer.id = sg.Tensor_sig.id; sg; role }

(* Every kernel input a caller binds is a buffer, used or not: [Kernel_eval]
   reports an unbound or mismatched input before evaluating anything, in input
   order, and the program's buffer order is what makes the interpreter do the
   same. A [Filled] input is a constant and has none. *)
let input_sources (k : Kernel.t) =
  List.fold_left
    (fun (buffers, sources) (i : Kernel.Input.t) ->
      let sg = i.Kernel.Input.sg in
      match i.Kernel.Input.binding with
      | Kernel.Binding.Caller | Kernel.Binding.Captured_constant ->
          let b = buffer sg Loop_buffer.Input in
          ( b :: buffers,
            Tensor_id.Map.add sg.Tensor_sig.id (Loop_lower_ctx.Buffer b) sources
          )
      | Kernel.Binding.Filled v ->
          ( buffers,
            Tensor_id.Map.add sg.Tensor_sig.id
              (Loop_lower_ctx.Fill (buffer sg Loop_buffer.Scratch, v))
              sources )
      | Kernel.Binding.Filled_i64 v ->
          ( buffers,
            Tensor_id.Map.add sg.Tensor_sig.id
              (Loop_lower_ctx.Fill_i64 (buffer sg Loop_buffer.Scratch, v))
              sources ))
    ([], Tensor_id.Map.empty) k.Kernel.inputs

(* The dense six-axis nest, N outermost and C innermost, as [Vec6.iter] visits a
   tensor. [lower] produces the stored value at the cell, inside the nest. *)
let nest_with ctx ~id ~(sg : Tensor_sig.t) lower =
  let vars =
    List.map (fun a -> (a, Loop_lower_ctx.fresh_var ctx)) Expr.Axis.all
  in
  let extent a = Dim.to_int (Vec6.get sg.Tensor_sig.shape a) in
  let ctx =
    {
      ctx with
      Loop_lower_ctx.at = id;
      axes = Expr.Coord.of_fn (fun a -> Loop_index.Var (List.assoc a vars));
      ranges =
        List.fold_left
          (fun env (a, var) ->
            Loop_range.Env.add_var var
              (Loop_range.span ~lo:0 ~hi:(extent a))
              env)
          ctx.Loop_lower_ctx.ranges vars;
      meter = ref false;
      block = ref [];
    }
  in
  let value = lower ctx in
  Loop_lower_ctx.emit ctx
    (Loop_stmt.Store
       {
         buffer = buffer sg Loop_buffer.Output;
         coord = ctx.Loop_lower_ctx.axes;
         value;
       });
  List.fold_right
    (fun a inner ->
      [
        Loop_stmt.For
          {
            var = List.assoc a vars;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const (extent a);
            body = inner;
          };
      ])
    Expr.Axis.all
    ((if !(ctx.Loop_lower_ctx.meter) then [ Loop_stmt.Reset_meter ] else [])
    @ List.rev !(ctx.Loop_lower_ctx.block))

let nest ctx (v : Kernel.Value.t) body =
  nest_with ctx ~id:v.Kernel.Value.id ~sg:v.Kernel.Value.sg (fun ctx ->
      Loop_lower_ctx.stored_of v
        (Loop_lower_value.value ctx
           (Kernel.Result_conversion.apply v.Kernel.Value.result body)))

(* An int64 value is exact end to end: no conversion, and stored as int64. *)
let nest_i64 ctx (v : Kernel.Value_i64.t) =
  nest_with ctx ~id:v.Kernel.Value_i64.id ~sg:v.Kernel.Value_i64.sg (fun ctx ->
      Loop_stored.I64 (Loop_lower_value.value_i64 ctx v.Kernel.Value_i64.pixel))

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
      | Error _ -> Loop_lower_ctx.refuse ctx Loop_unsupported.Virtual_use)
  | _ :: _ :: _ -> Loop_lower_ctx.refuse ctx Loop_unsupported.Virtual_use

(* A stored Solo Region value: the format check a Pixel nest makes, then the
   key loop of [Loop_lower_region]. *)
let region_unit ctx (v : Kernel.Value.t) program ~limits =
  (* The reference rejects a Region program over its admission budget (local
     slots, scan state, updates per key) before it evaluates anything, by
     [Region_program.preflight] on the converted program. Refusing what it rejects
     keeps the two from disagreeing about a program neither is meant to run. *)
  (match
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
  | Error _ -> Loop_lower_ctx.refuse ctx Loop_unsupported.Region_admission);
  Loop_lower_region.lower
    { ctx with Loop_lower_ctx.meter = ref false }
    v program

(* A run of grouped values: one shared recurrence per canonical key, one store
   per SELECTED member. The reference re-validates each member's converted
   emitter before running, and so does this. *)
let group_unit ctx g selected ~limits =
  let converted =
    Region_group.map_outputs g (fun ordinal output ->
        match List.assoc_opt ordinal selected with
        | Some (v : Kernel.Value.t) ->
            Kernel.Result_conversion.apply v.Kernel.Value.result output
        | None -> output)
  in
  (match
     Region_execution.lower_group ~max_size:limits.Kernel.Limits.max_size
       ~max_depth:limits.Kernel.Limits.max_depth
       ~max_local_slots:limits.Kernel.Limits.max_local_slots
       ~scan_limits:(Kernel.Limits.scan_limits limits)
       converted
   with
  | Ok _ -> ()
  | Error _ -> Loop_lower_ctx.refuse ctx Loop_unsupported.Region_admission);
  Loop_lower_region.lower_group ctx g selected

let lower (plan : Fusion_plan.t) =
  Err.Escape.with_escape @@ fun esc ->
  let k = plan.Fusion_plan.kernel in
  let stored (v : Kernel.Value.t) =
    Tensor_id.Set.mem v.Kernel.Value.id plan.Fusion_plan.stores
  in
  let inputs, sources = input_sources k in
  let outputs =
    List.filter_map
      (fun (v : Kernel.Value.t) ->
        if stored v then Some (buffer v.Kernel.Value.sg Loop_buffer.Output)
        else None)
      k.Kernel.values
  in
  let i64_outputs =
    List.map
      (fun (v : Kernel.Value_i64.t) ->
        buffer v.Kernel.Value_i64.sg Loop_buffer.Output)
      k.Kernel.values_i64
  in
  let sources =
    List.fold_left
      (fun m (b : Loop_buffer.t) ->
        Tensor_id.Map.add b.Loop_buffer.id (Loop_lower_ctx.Buffer b) m)
      sources (outputs @ i64_outputs)
  in
  let ctx =
    {
      Loop_lower_ctx.esc;
      at = Tensor_id.of_int 0;
      sources;
      axes = Expr.Coord.of_fn (fun _ -> Loop_index.Const 0);
      reducers = Expr.Reduce_var.Map.empty;
      locals = Expr.Local_var.Map.empty;
      ranges = Loop_range.Env.create ();
      supply =
        {
          Loop_lower_ctx.arrays = Loop_array.Next.first;
          vars = Loop_var.Next.first;
          temps = Loop_temp.Next.first;
        };
      meter = ref false;
      hoisted = ref [];
      block = ref [];
    }
  in
  (* An int64 value runs before the first float value that reads it, and after
     everything it reads: [Kernel.create] checked the two lists as one dependency
     graph, so an entry reads only earlier floats and earlier entries. The rest
     follow the floats, in list order. [Kernel_eval] instead produces one on
     demand, when the first load of it is evaluated; both fail with the same row,
     but a float value that would fail before reaching such a load fails first
     there and after it here. *)
  let i64_entries = k.Kernel.values_i64 in
  let emitted = ref Tensor_id.Set.empty in
  let rec emit_i64 (v : Kernel.Value_i64.t) =
    if Tensor_id.Set.mem v.Kernel.Value_i64.id !emitted then []
    else (
      emitted := Tensor_id.Set.add v.Kernel.Value_i64.id !emitted;
      let deps =
        Expr.Source.Set.fold
          (fun src acc ->
            match
              Tensor_id.Map.find_opt
                (Expr_bridge.id_of_source src)
                k.Kernel.by_id_i64
            with
            | Some dep -> acc @ emit_i64 dep
            | None -> acc)
          (Expr.Fold.sources_i64 v.Kernel.Value_i64.pixel)
          []
      in
      deps @ nest_i64 { ctx with Loop_lower_ctx.at = v.Kernel.Value_i64.id } v)
  in
  let reads_i64 sources =
    List.concat_map
      (fun (v : Kernel.Value_i64.t) ->
        if
          Expr.Source.Set.mem
            (Expr_bridge.source_of_id v.Kernel.Value_i64.id)
            sources
        then emit_i64 v
        else [])
      i64_entries
  in
  let runs =
    Region_group.runs
      ~computation:(fun (v : Kernel.Value.t) -> v.Kernel.Value.computation)
      k.Kernel.values
  in
  let float_units =
    List.concat_map
      (function
        | Region_group.Run.Solo v when stored v -> (
            reads_i64 (Region_group.Ref.sources v.Kernel.Value.computation)
            @
            match
              Region_group.Ref.pixel_expression v.Kernel.Value.computation
            with
            | Some pixel ->
                let ctx = { ctx with Loop_lower_ctx.at = v.Kernel.Value.id } in
                nest ctx v (body ctx plan v pixel)
            | None -> (
                let ctx = { ctx with Loop_lower_ctx.at = v.Kernel.Value.id } in
                let limits = k.Kernel.limits in
                match
                  Region_group.Ref.project
                    ~max_size:limits.Kernel.Limits.max_size
                    ~max_depth:limits.Kernel.Limits.max_depth
                    v.Kernel.Value.computation
                with
                | Ok program -> region_unit ctx v program ~limits
                | Error _ ->
                    Loop_lower_ctx.refuse ctx Loop_unsupported.Region_program))
        | Region_group.Run.Solo _ -> []
        | Region_group.Run.Group (g, members) -> (
            match List.filter (fun (_, v) -> stored v) members with
            | [] -> []
            | (_, first) :: _ as selected ->
                reads_i64
                  (List.fold_left
                     (fun acc (_, (v : Kernel.Value.t)) ->
                       Expr.Source.Set.union acc
                         (Region_group.Ref.sources v.Kernel.Value.computation))
                     Expr.Source.Set.empty selected)
                @
                let ctx =
                  {
                    ctx with
                    Loop_lower_ctx.at = first.Kernel.Value.id;
                    meter = ref false;
                  }
                in
                group_unit ctx g selected ~limits:k.Kernel.limits))
      runs
  in
  (* Sequenced with a [let]: an operand of [@] is evaluated right to left, and the
     leftover int64 entries must be emitted only after every unit above has asked
     for the ones it reads. *)
  let statements = float_units @ List.concat_map emit_i64 i64_entries in
  {
    Loop_program.buffers = List.rev inputs @ outputs @ i64_outputs;
    body = List.rev !(ctx.Loop_lower_ctx.hoisted) @ statements;
    scan_limits = Kernel.Limits.scan_limits k.Kernel.limits;
    max_depth = k.Kernel.limits.Kernel.Limits.max_depth;
  }
