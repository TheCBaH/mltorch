type block = {
  id : Mir_id.Block.t;
  params : Mir_value.t list;
  order0 : Mir_value.t;
  mutable current : Mir_value.t;
  mutable body : Mir_op.t Mir_instr.t list;  (** reversed *)
  mutable terminator : Mir_terminator.t option;
}

type t = {
  mutable next_value : Mir_id.Value.Next.t;
  mutable next_block : Mir_id.Block.Next.t;
  mutable next_instr : Mir_id.Instr.Next.t;
  mutable blocks : block list;  (** since the last [func], reversed *)
}

let create () =
  {
    next_value = Mir_id.Value.Next.first;
    next_block = Mir_id.Block.Next.first;
    next_instr = Mir_id.Instr.Next.first;
    blocks = [];
  }

let block_id b = b.id
let param b = b.params
let order b = b.current

let value t ty =
  Mir_id.Value.Next.check_room t.next_value ~count:1;
  let id, next = Mir_id.Value.Next.alloc t.next_value in
  t.next_value <- next;
  { Mir_value.id; ty }

let new_block t tys =
  Mir_id.Block.Next.check_room t.next_block ~count:1;
  let id, next = Mir_id.Block.Next.alloc t.next_block in
  t.next_block <- next;
  let params = List.map (value t) tys in
  let order0 = value t Mir_type.Order in
  let b =
    { id; params; order0; current = order0; body = []; terminator = None }
  in
  t.blocks <- b :: t.blocks;
  b

let op ?(origin = Mir_origin.unknown) t b ~signature op =
  match Mir_typing.check ~signature ~view:(fun _ -> true) op with
  | Error e -> Error e
  | Ok tys ->
      if Option.is_some b.terminator then
        invalid_arg "Mir_builder: block already terminated";
      Mir_id.Instr.Next.check_room t.next_instr ~count:1;
      let id, next = Mir_id.Instr.Next.alloc t.next_instr in
      t.next_instr <- next;
      let results = List.map (value t) tys in
      let order =
        if Mir_op.Effect.ordered (Mir_op.effect_class op) then (
          let output = value t Mir_type.Order in
          let o = { Mir_order.input = b.current; output } in
          b.current <- output;
          Some o)
        else None
      in
      b.body <- { Mir_instr.id; results; op; order; origin } :: b.body;
      Ok results

let no_signature _ = None

let emit ?origin t b o =
  match op ?origin t b ~signature:no_signature o with
  | Ok [ r ] -> r
  | Ok _ -> invalid_arg "Mir_builder.emit: not exactly one result"
  | Error e ->
      invalid_arg (Fmt.str "Mir_builder.emit: %a" Mir_typing.Error.pp e)

let emit_unit ?origin t b o =
  match op ?origin t b ~signature:no_signature o with
  | Ok [] -> ()
  | Ok _ -> invalid_arg "Mir_builder.emit_unit: a result"
  | Error e ->
      invalid_arg (Fmt.str "Mir_builder.emit_unit: %a" Mir_typing.Error.pp e)

let terminate b term =
  if Option.is_some b.terminator then
    invalid_arg "Mir_builder: block already terminated";
  b.terminator <- Some term

let edge from (target, args) =
  { Mir_edge.target = target.id; args; order = from.current }

let jump b target args =
  terminate b (Mir_terminator.Jump (edge b (target, args)))

let branch b cond then_ else_ =
  terminate b
    (Mir_terminator.Branch
       { Mir_branch.cond; then_ = edge b then_; else_ = edge b else_ })

let return b values =
  terminate b (Mir_terminator.Return { Mir_return.values; order = b.current })

let fail b failure payload =
  terminate b
    (Mir_terminator.Fail { Mir_fail.failure; payload; order = b.current })

let finish b =
  match b.terminator with
  | None -> invalid_arg "Mir_builder.func: a block has no terminator"
  | Some terminator ->
      {
        Mir_block.id = b.id;
        params = b.params;
        order = b.order0;
        body = List.rev b.body;
        terminator;
      }

let func t ~id ~name ~entry ~results =
  let blocks = List.rev t.blocks in
  t.blocks <- [];
  let entry_block = finish entry in
  let rest =
    List.filter_map
      (fun b ->
        if Mir_id.Block.equal b.id entry.id then None else Some (finish b))
      blocks
  in
  { Mir_func.id; name; entry = entry.id; results; blocks = entry_block :: rest }

let program ?(regions = []) ?(views = []) ?(helpers = []) ?planning funcs ~main
    =
  {
    Mir_program.data_model = Mir_layout.Data_model.Lp64_le;
    regions;
    views;
    helpers;
    funcs;
    main;
    planning;
    revision = Mir_id.Revision.of_int 0;
  }
