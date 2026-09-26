(* Lowering a Region program in [Region_execution.materialize]'s order, which is
   the definition of when each local is evaluated:

     alloc slots : float64[Region_slots total]        once, reused across keys
     for each key over the partition's Singleton axes:
       for each local, in declared order: write its slots
       for each output over the Whole axes: store conv(emitter)

   A local runs once per KEY, never once per output; sinking one into the
   emitter loop would be a recomputation, not an equivalent schedule. Locals hold
   working binary64, so the result conversion is spliced into the emitter only.

   The slot array is reused across keys rather than zeroed: locals are ordered
   ([Region_program.check] rejects a read of a later local) and every slot a body
   reads is written earlier in the same key, so a stale value from the previous
   key is never observable. *)

open Loop_lower_ctx

let wrap_loops ~extent vars inner =
  List.fold_right
    (fun (a, var) body ->
      [
        Loop_stmt.For
          {
            var;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const (extent a);
            body;
          };
      ])
    vars inner

(* A trace local: row 0 is the initializer, one evaluation per lane; rows
   [1..steps] are the update, charged once per lane BEFORE its body, with [prev]
   reading row [step] straight from the slots just written. The meter is the
   key's, shared with every other local and the emitter. *)
let write_trace ctx ~array ~offset (s : Expr.Scan.t) =
  let width = s.Expr.Scan.width and steps = s.Expr.Scan.steps in
  ctx.meter := true;
  emit ctx (Loop_stmt.Mark Loop_mark.Scan);
  let lane_range = Loop_range.span ~lo:0 ~hi:width in
  let cell ~base l =
    (* [offset + base + l]: the flat slot of a lane in a row. *)
    Loop_index.Add (Loop_index.Add (Loop_index.Const offset, base), l)
  in
  let each_lane ctx body =
    let l = fresh_var ctx in
    let inner =
      {
        ctx with
        reducers =
          Expr.Reduce_var.Map.add s.Expr.Scan.lane (Loop_index.Var l)
            ctx.reducers;
        ranges = Loop_range.Env.add_var l lane_range ctx.ranges;
      }
    in
    let (), stmts =
      in_block inner (fun inner -> body inner (Loop_index.Var l))
    in
    Loop_stmt.For
      {
        var = l;
        lo = Loop_index.Const 0;
        hi = Loop_index.Const width;
        body = stmts;
      }
  in
  emit ctx
    (each_lane ctx (fun inner l ->
         let e = Loop_lower_value.value inner s.Expr.Scan.init in
         emit inner
           (Loop_stmt.Array_set (array, cell ~base:(Loop_index.Const 0) l, e));
         emit inner (Loop_stmt.Mark Loop_mark.Local)));
  let step = fresh_var ctx in
  let stepper =
    {
      ctx with
      reducers =
        Expr.Reduce_var.Map.add s.Expr.Scan.step (Loop_index.Var step)
          ctx.reducers;
      ranges =
        Loop_range.Env.add_var step (Loop_range.span ~lo:0 ~hi:steps) ctx.ranges;
      locals =
        Expr.Local_var.Map.add s.Expr.Scan.prev
          (Prev_row
             {
               array;
               base =
                 Loop_index.Add
                   ( Loop_index.Const offset,
                     Loop_index.Scale (width, Loop_index.Var step) );
               width;
             })
          ctx.locals;
    }
  in
  let (), rows =
    in_block stepper (fun stepper ->
        emit stepper
          (each_lane stepper (fun inner l ->
               emit inner Loop_stmt.Charge_scan_update;
               emit inner (Loop_stmt.Mark Loop_mark.Scan_update);
               let e = Loop_lower_value.value inner s.Expr.Scan.update in
               let row =
                 Loop_index.Scale
                   ( width,
                     Loop_index.Add (Loop_index.Var step, Loop_index.Const 1) )
               in
               emit inner (Loop_stmt.Array_set (array, cell ~base:row l, e));
               emit inner (Loop_stmt.Mark Loop_mark.Local))))
  in
  emit ctx
    (Loop_stmt.For
       {
         var = step;
         lo = Loop_index.Const 0;
         hi = Loop_index.Const steps;
         body = rows;
       })

(* One local's slots, written into the key's block. Returns the binding a LATER
   local or the emitter reads it through. *)
let write_local ctx ~array ~slots (l : Region_local.t) =
  let range = Option.get (Region_slots.offset slots l.Region_local.id) in
  let offset = (range.Slot.Range.offset :> int) in
  let shape = Region_local.Shape.of_rhs l.Region_local.rhs in
  (match l.Region_local.rhs with
  | Region_local.Rhs.Scalar body ->
      let e = Loop_lower_value.value ctx body in
      emit ctx (Loop_stmt.Array_set (array, Loop_index.Const offset, e));
      emit ctx (Loop_stmt.Mark Loop_mark.Local)
  | Region_local.Rhs.Vector { extent; var; body } ->
      (* The body is evaluated once per position with its own binder bound to
         the position, the loop shape [Reduce]'s fold uses. *)
      let extent = (extent :> int) in
      let p = fresh_var ctx in
      let inner =
        {
          ctx with
          reducers = Expr.Reduce_var.Map.add var (Loop_index.Var p) ctx.reducers;
          ranges =
            Loop_range.Env.add_var p
              (Loop_range.span ~lo:0 ~hi:extent)
              ctx.ranges;
        }
      in
      let (), body =
        in_block inner (fun inner ->
            let e = Loop_lower_value.value inner body in
            emit inner
              (Loop_stmt.Array_set
                 ( array,
                   (if offset = 0 then Loop_index.Var p
                    else
                      Loop_index.Add (Loop_index.Const offset, Loop_index.Var p)),
                   e ));
            emit inner (Loop_stmt.Mark Loop_mark.Local))
      in
      emit ctx
        (Loop_stmt.For
           {
             var = p;
             lo = Loop_index.Const 0;
             hi = Loop_index.Const extent;
             body;
           })
  | Region_local.Rhs.Scan s -> write_trace ctx ~array ~offset s);
  Slots { array; range; shape }

(* One output of a Region computation: the value it stores, the shape and
   partition it is stored under, how its physical key is read off the canonical
   key, and its emitter expression with the value's own result conversion already
   applied. A solo Region value has one of these and an identity key mapping; a
   group has one per member. *)
type emitter = {
  value : Kernel.Value.t;
  output_shape : Vec6.shape;
  partition : Region_partition.t;
  key_axes : (Expr.Axis.t * Expr.Axis.t) list;
      (** (canonical axis, physical axis): the physical Singleton axis takes the
          canonical key's value; an axis no pair names reads 0 *)
  output : float Expr.Value.t;
}

let singleton partition a =
  Region_partition.mode partition a = Region_partition.Axis_mode.Singleton

(* One nest over the canonical key: the shared locals once per key, then each
   emitter in order, each over its own Whole axes at its own physical key. Every
   emitter carries its own conversion and its own store: leaving the conversion to
   the store is not equivalent (an f32 round before a Bool member's nonzero test
   reads a working value below binary32's range as false). *)
let lower_nest ctx ~canonical_shape ~canonical_partition ~locals ~emitters =
  let extent_of shape a = Dim.to_int (Vec6.get shape a) in
  let slots = Region_slots.of_locals locals in
  let array = fresh_array ctx in
  let key_vars =
    List.filter_map
      (fun a ->
        if singleton canonical_partition a then Some (a, fresh_var ctx)
        else None)
      Expr.Axis.all
  in
  let var_of vars a = Loop_index.Var (List.assoc a vars) in
  let ranges =
    List.fold_left
      (fun env (a, var) ->
        Loop_range.Env.add_var var
          (Loop_range.span ~lo:0 ~hi:(extent_of canonical_shape a))
          env)
      ctx.ranges key_vars
  in
  (* Inside a local, [Output a] reads the canonical key: the loop variable of a
     Singleton axis and 0 for a Whole one. [Region_program.check]'s
     [Non_invariant_local] rule proves no local reads a Whole axis, so the 0 is
     never observed. *)
  let key_coord =
    Expr.Coord.of_fn (fun a ->
        if singleton canonical_partition a then var_of key_vars a
        else Loop_index.Const 0)
  in
  let key_ctx = { ctx with axes = key_coord; ranges; block = ref [] } in
  emit key_ctx (Loop_stmt.Mark Loop_mark.Key);
  let bound =
    List.fold_left
      (fun key_ctx l ->
        let binding = write_local key_ctx ~array ~slots l in
        {
          key_ctx with
          locals =
            Expr.Local_var.Map.add l.Region_local.id binding key_ctx.locals;
        })
      key_ctx locals
  in
  let emit_one (e : emitter) =
    let out_vars =
      List.filter_map
        (fun a ->
          if singleton e.partition a then None else Some (a, fresh_var ctx))
        Expr.Axis.all
    in
    let ranges =
      List.fold_left
        (fun env (a, var) ->
          Loop_range.Env.add_var var
            (Loop_range.span ~lo:0 ~hi:(extent_of e.output_shape a))
            env)
        bound.ranges out_vars
    in
    (* A Singleton axis reads the physical key (the canonical key's value at the
       axis a pair maps to it, else the origin), a Whole axis its own loop. *)
    let coord =
      Expr.Coord.of_fn (fun a ->
          if singleton e.partition a then
            match List.find_opt (fun (_, phys) -> phys = a) e.key_axes with
            | Some (canon, _) -> var_of key_vars canon
            | None -> Loop_index.Const 0
          else var_of out_vars a)
    in
    let emitter_ctx =
      {
        bound with
        at = e.value.Kernel.Value.id;
        axes = coord;
        ranges;
        block = ref [];
      }
    in
    emit emitter_ctx (Loop_stmt.Mark Loop_mark.Emitter);
    let x = Loop_lower_value.value emitter_ctx e.output in
    emit emitter_ctx
      (Loop_stmt.Store
         {
           buffer = output_buffer e.value.Kernel.Value.sg;
           coord;
           value = stored_of e.value x;
         });
    wrap_loops ~extent:(extent_of e.output_shape) out_vars
      (List.rev !(emitter_ctx.block))
  in
  let outputs = List.concat_map emit_one emitters in
  (* One fresh meter per canonical key, shared by every local (a scan's own trace
     fill included) and every emitter visiting that key: the reset scope the scan
     design specifies for Region materialization. *)
  let key_stmts =
    match List.rev !(key_ctx.block) with
    | key :: rest when !(ctx.meter) -> key :: Loop_stmt.Reset_meter :: rest
    | stmts -> stmts
  in
  let keys =
    wrap_loops ~extent:(extent_of canonical_shape) key_vars (key_stmts @ outputs)
  in
  (if (Region_slots.total slots :> int) > 0 then
     [ Loop_stmt.Alloc (array, Region_slots.total slots) ]
   else [])
  @ keys

(* A solo value: its own partition is the canonical one and every Singleton axis
   maps to itself. *)
let lower ctx (v : Kernel.Value.t) (program : Region_program.t) =
  let shape = v.Kernel.Value.sg.Tensor_sig.shape in
  let partition = Region_program.partition program in
  lower_nest
    { ctx with at = v.Kernel.Value.id }
    ~canonical_shape:shape ~canonical_partition:partition
    ~locals:(Region_program.locals program)
    ~emitters:
      [
        {
          value = v;
          output_shape = shape;
          partition;
          key_axes =
            List.filter_map
              (fun a -> if singleton partition a then Some (a, a) else None)
              Expr.Axis.all;
          output =
            Kernel.Result_conversion.apply v.Kernel.Value.result
              (Region_program.output program);
        };
      ]

(* A run of grouped values, the selected members in run order. *)
let lower_group ctx (g : Region_group.t)
    (members : (Region_group.Ordinal.t * Kernel.Value.t) list) =
  lower_nest ctx
    ~canonical_shape:(Region_group.canonical_shape g)
    ~canonical_partition:(Region_group.canonical_partition g)
    ~locals:(Region_group.locals g)
    ~emitters:
      (List.map
         (fun (ordinal, (v : Kernel.Value.t)) ->
           let e = Option.get (Region_group.emitter g ordinal) in
           {
             value = v;
             output_shape = e.Region_group.Emitter.output_shape;
             partition = e.Region_group.Emitter.partition;
             key_axes = e.Region_group.Emitter.key_axes;
             output =
               Kernel.Result_conversion.apply v.Kernel.Value.result
                 e.Region_group.Emitter.output;
           })
         members)
