open Ssa_ir
open Ssa_lower_ctx
module B = Ssa_builder

(* Lowering a Region program in [Region_execution.materialize]'s order, which is
   the definition of when each local is evaluated:

     for each key over the partition's Singleton axes:
       meter reset, mark key
       for each local, in declared order: allocate it, write its slots
       for each output over the Whole axes: store conv(emitter)

   A local runs once per KEY, never once per output; sinking one into the
   emitter loop would be a recomputation, not an equivalent schedule. Each local
   is a scratch object allocated inside the key's region, so what one key wrote
   is never visible to the next. Locals hold working binary64, so the result
   conversion is spliced into the emitter only. *)

let extent_of shape a = Dim.to_int (Vec6.get shape a)
let zero b = B.index b 0L
let const b n = B.index b (Int64.of_int n)

(* A loop over [0, extent) with a body that returns nothing. *)
let count_loop b ~extent body =
  let B.Nil =
    B.for_ b ~lo:(zero b) ~hi:(const b extent) ~init:B.Nil (fun b i B.Nil ->
        body b i;
        B.Nil)
  in
  ()

(* A trace local: row 0 is the initializer, one evaluation per lane; rows
   [1..steps] are the update, charged once per lane BEFORE its body, with [prev]
   reading row [step] straight from the object just written. The meter is the
   key's, shared with every other local and the emitter. *)
let write_trace ctx ~handle (s : Expr.Scan.t) =
  let width = s.Expr.Scan.width and steps = s.Expr.Scan.steps in
  ctx.meter := true;
  B.mark ctx.b Ssa_mark.Scan;
  count_loop ctx.b ~extent:width (fun b l ->
      let inner =
        {
          ctx with
          b;
          reducers = Expr.Reduce_var.Map.add s.Expr.Scan.lane l ctx.reducers;
        }
      in
      B.local_write b handle l (Ssa_lower_value.value inner s.Expr.Scan.init);
      B.mark b Ssa_mark.Local);
  count_loop ctx.b ~extent:steps (fun b step ->
      let row = B.index_scale b (Int64.of_int width) step in
      let next =
        B.index_scale b (Int64.of_int width) (B.index_add b step (const b 1))
      in
      count_loop b ~extent:width (fun b l ->
          B.meter_charge b;
          B.mark b Ssa_mark.Scan_update;
          let inner =
            {
              ctx with
              b;
              reducers =
                Expr.Reduce_var.Map.add s.Expr.Scan.lane l
                  (Expr.Reduce_var.Map.add s.Expr.Scan.step step ctx.reducers);
              locals =
                Expr.Local_var.Map.add s.Expr.Scan.prev
                  (Prev_row { handle; base = row; width })
                  ctx.locals;
            }
          in
          let x = Ssa_lower_value.value inner s.Expr.Scan.update in
          B.local_write b handle (B.index_add b next l) x;
          B.mark b Ssa_mark.Local))

(* One local's object, written into the key's block. Returns the binding a LATER
   local or the emitter reads it through. *)
let write_local ctx (l : Region_local.t) =
  let count = (Region_local.Rhs.slot_count l.Region_local.rhs :> int) in
  let shape = Region_local.Shape.of_rhs l.Region_local.rhs in
  let handle =
    B.local_alloc ~var:l.Region_local.id ctx.b ~slots:(Int64.of_int count)
  in
  (match l.Region_local.rhs with
  | Region_local.Rhs.Scalar body ->
      let x = Ssa_lower_value.value ctx body in
      B.local_write ctx.b handle (zero ctx.b) x;
      B.mark ctx.b Ssa_mark.Local
  | Region_local.Rhs.Vector { extent; var; body } ->
      (* The body is evaluated once per position with its own binder bound to
         the position, the loop shape [Reduce]'s fold uses. *)
      count_loop ctx.b
        ~extent:(extent :> int)
        (fun b p ->
          let inner =
            {
              ctx with
              b;
              reducers = Expr.Reduce_var.Map.add var p ctx.reducers;
            }
          in
          B.local_write b handle p (Ssa_lower_value.value inner body);
          B.mark b Ssa_mark.Local)
  | Region_local.Rhs.Scan s -> write_trace ctx ~handle s);
  Slots { handle; count; shape }

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

(* [for a in 0..extent] over each axis of [axes], outermost first, then [body]
   with the induction value of each. *)
let rec nest b ~extent axes inductions body =
  match axes with
  | [] -> body b (List.rev inductions)
  | a :: rest ->
      count_loop b ~extent:(extent a) (fun b i ->
          nest b ~extent rest ((a, i) :: inductions) body)

(* One nest over the canonical key: the shared locals once per key, then each
   emitter in order, each over its own Whole axes at its own physical key. Every
   emitter carries its own conversion and its own store: leaving the conversion to
   the store is not equivalent (an f32 round before a Bool member's nonzero test
   reads a working value below binary32's range as false). *)
let lower_nest ctx ~canonical_shape ~canonical_partition ~locals ~emitters =
  let key_axes =
    List.filter (fun a -> singleton canonical_partition a) Expr.Axis.all
  in
  (* One key: the meter (fresh for the key, shared by every local and emitter
     visiting it), the shared locals once, then each emitter. [reset] says
     whether the key reads the meter at all, which only lowering it can tell. *)
  let key_body ~meter ~reset b key =
    let key_value a = List.assoc a key in
    (* Inside a local, [Output a] reads the canonical key: the loop variable of a
       Singleton axis and 0 for a Whole one. [Region_program.check]'s
       [Non_invariant_local] rule proves no local reads a Whole axis, so the 0 is
       never observed. *)
    let components =
      List.map
        (fun a ->
          (a, if singleton canonical_partition a then key_value a else zero b))
        Expr.Axis.all
    in
    let key_ctx =
      {
        ctx with
        b;
        axes = Some (Expr.Coord.of_fn (fun a -> List.assoc a components));
        locals = Expr.Local_var.Map.empty;
        meter;
      }
    in
    B.mark b Ssa_mark.Key;
    if reset then B.meter_reset b;
    let bound =
      List.fold_left
        (fun key_ctx (l : Region_local.t) ->
          let binding = write_local key_ctx l in
          {
            key_ctx with
            locals =
              Expr.Local_var.Map.add l.Region_local.id binding key_ctx.locals;
          })
        key_ctx locals
    in
    List.iter
      (fun (e : emitter) ->
        let whole =
          List.filter (fun a -> not (singleton e.partition a)) Expr.Axis.all
        in
        nest b ~extent:(extent_of e.output_shape) whole [] (fun b out ->
            (* A Singleton axis reads the physical key (the canonical key's value
               at the axis a pair maps to it, else the origin), a Whole axis its
               own loop. *)
            let coord_components =
              List.map
                (fun a ->
                  ( a,
                    if singleton e.partition a then
                      match
                        List.find_opt (fun (_, phys) -> phys = a) e.key_axes
                      with
                      | Some (canon, _) -> key_value canon
                      | None -> zero b
                    else List.assoc a out ))
                Expr.Axis.all
            in
            let coord =
              Expr.Coord.of_fn (fun a -> List.assoc a coord_components)
            in
            let emitter_ctx =
              { bound with b; at = e.value.Kernel.Value.id; axes = Some coord }
            in
            B.mark b Ssa_mark.Emitter;
            let x = Ssa_lower_value.value emitter_ctx e.output in
            B.store_f64 b
              (Ssa_lower_ctx.buffer_id e.value.Kernel.Value.id)
              ~encode:
                (match e.value.Kernel.Value.result with
                | Kernel.Result_conversion.Nonzero_bool ->
                    Ssa_op.Encode.Bool_nonzero
                | Kernel.Result_conversion.Round_f32 -> Ssa_op.Encode.F32_round)
              (B.Coord coord) x))
      emitters
  in
  let uses_meter =
    let probed = ref false in
    B.probe ctx.b (fun b ->
        let key = List.map (fun a -> (a, zero b)) key_axes in
        let meter = ref false in
        key_body ~meter ~reset:false b key;
        probed := !meter);
    !probed
  in
  nest ctx.b ~extent:(extent_of canonical_shape) key_axes [] (fun b key ->
      key_body ~meter:(ref false) ~reset:uses_meter b key)

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
