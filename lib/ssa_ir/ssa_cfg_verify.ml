type problem =
  | Block_defined_twice
  | Branch_condition of Ssa_type.t
  | Buffer_format of {
      buffer : Ssa_id.Buffer.t;
      accessed : Ssa_format.Family.t;
      declared : Ssa_format.t;
    }
  | Buffer_not_stored of Ssa_id.Buffer.t
  | Buffer_unknown of Ssa_id.Buffer.t
  | Definition_twice of Ssa_id.Value.t
  | Edge_arguments of { expected : Ssa_type.t list; found : Ssa_type.t list }
  | Effect_missing
  | Effect_parameters of Ssa_type.t list
  | Effect_stale of { used : Ssa_value.t; live : Ssa_value.t }
  | Entry_has_predecessor
  | Entry_missing
  | Flat_on_per_channel of Ssa_id.Buffer.t
  | Result_types of { declared : Ssa_type.t list; expected : Ssa_type.t list }
  | Target_unknown of Ssa_id.Block.t
  | Typing of Ssa_typing.error
  | Unreachable
  | Use_not_dominated of Ssa_id.Value.t
  | Use_retyped of {
      value : Ssa_id.Value.t;
      defined : Ssa_type.t;
      used : Ssa_type.t;
    }
  | Use_undefined of Ssa_id.Value.t

type diagnostic = { block : Ssa_id.Block.t; problem : problem }
type error = [ `Invalid_cfg of diagnostic ]

let pp_types = Fmt.list ~sep:(Fmt.any ", ") Ssa_type.pp

let pp_problem fmt = function
  | Block_defined_twice -> Fmt.string fmt "the block appears twice"
  | Branch_condition t ->
      Fmt.pf fmt "a branch condition is a predicate, not %a" Ssa_type.pp t
  | Buffer_format { buffer; accessed; declared } ->
      Fmt.pf fmt "%a is declared %s but accessed as %s" Ssa_id.Buffer.pp buffer
        (Ssa_format.name declared)
        (Ssa_format.Family.name accessed)
  | Buffer_not_stored b ->
      Fmt.pf fmt "%a is an input and is never written" Ssa_id.Buffer.pp b
  | Buffer_unknown b -> Fmt.pf fmt "%a is not declared" Ssa_id.Buffer.pp b
  | Definition_twice v -> Fmt.pf fmt "%a is defined twice" Ssa_id.Value.pp v
  | Edge_arguments { expected; found } ->
      Fmt.pf fmt "an edge passes (%a) to parameters (%a)" pp_types found
        pp_types expected
  | Effect_missing -> Fmt.string fmt "effect operand missing or unexpected"
  | Effect_parameters ts ->
      Fmt.pf fmt
        "a block has one effect parameter, or none and one predecessor; \
         parameters (%a)"
        pp_types ts
  | Effect_stale { used; live } ->
      Fmt.pf fmt "effect %a is not the current effect %a" Ssa_id.Value.pp
        used.Ssa_value.id Ssa_id.Value.pp live.Ssa_value.id
  | Entry_has_predecessor -> Fmt.string fmt "the entry block has a predecessor"
  | Entry_missing -> Fmt.string fmt "the entry block is not in the graph"
  | Flat_on_per_channel b ->
      Fmt.pf fmt "%a is per-channel quantized and takes no flat access"
        Ssa_id.Buffer.pp b
  | Result_types { declared; expected } ->
      Fmt.pf fmt "results declared (%a), operation yields (%a)" pp_types
        declared pp_types expected
  | Target_unknown b -> Fmt.pf fmt "block %a does not exist" Ssa_id.Block.pp b
  | Typing e -> Ssa_typing.pp_error fmt e
  | Unreachable -> Fmt.string fmt "the block cannot be reached from the entry"
  | Use_not_dominated v ->
      Fmt.pf fmt "%a is not defined on every path to its use" Ssa_id.Value.pp v
  | Use_retyped { value; defined; used } ->
      Fmt.pf fmt "%a is defined as %a and used as %a" Ssa_id.Value.pp value
        Ssa_type.pp defined Ssa_type.pp used
  | Use_undefined v -> Fmt.pf fmt "%a is not defined" Ssa_id.Value.pp v

let pp_diagnostic fmt { block; problem } =
  Fmt.pf fmt "%a: %a" Ssa_id.Block.pp block pp_problem problem

let pp_error fmt : [< error ] -> unit = function
  | `Invalid_cfg d -> pp_diagnostic fmt d

type ctx = {
  esc : error Err.Escape.t;
  cfg : Ssa_cfg.t;
  idom : Ssa_id.Block.t Ssa_id.Block.Map.t;
  preds : Ssa_id.Block.t list Ssa_id.Block.Map.t;
  (* a value's definition: its block and where in it, parameters at -1 *)
  defs : (Ssa_id.Block.t * int * Ssa_type.t) Ssa_id.Value.Map.t;
}

let fail ctx block problem =
  Err.Escape.throw ctx.esc (`Invalid_cfg { block; problem } : error)

let types vs = List.map (fun (v : Ssa_value.t) -> v.Ssa_value.ty) vs
let is_effect (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect

(* A use at [position] of [block]: defined, of the type it is used at, and
   either earlier in the same block or in a block that dominates it. *)
let use ctx block ~position (v : Ssa_value.t) =
  match Ssa_id.Value.Map.find_opt v.Ssa_value.id ctx.defs with
  | None -> fail ctx block (Use_undefined v.Ssa_value.id)
  | Some (def_block, def_position, ty) ->
      if not (Ssa_type.equal ty v.Ssa_value.ty) then
        fail ctx block
          (Use_retyped
             { value = v.Ssa_value.id; defined = ty; used = v.Ssa_value.ty });
      let visible =
        if Ssa_id.Block.equal def_block block then def_position < position
        else Ssa_cfg.dominates ctx.cfg ctx.idom def_block block
      in
      if not visible then fail ctx block (Use_not_dominated v.Ssa_value.id)

let buffer_of ctx block id =
  match Ssa_cfg.find_buffer ctx.cfg id with
  | Some b -> b
  | None -> fail ctx block (Buffer_unknown id)

let check_access ctx block id ~family ~writes ~flat =
  let b = buffer_of ctx block id in
  if Ssa_format.family b.Ssa_buffer.format <> family then
    fail ctx block
      (Buffer_format
         { buffer = id; accessed = family; declared = b.Ssa_buffer.format });
  if flat && Ssa_format.per_channel b.Ssa_buffer.format then
    fail ctx block (Flat_on_per_channel id);
  if writes && b.Ssa_buffer.role = Ssa_buffer.Input then
    fail ctx block (Buffer_not_stored id)

let is_flat = function Ssa_access.Flat _ -> true | Ssa_access.Coord _ -> false

(* The buffer rules of one operation; its typing is checked by the caller. *)
let check_buffers ctx block (op : Ssa_op.t) =
  match op with
  | Ssa_op.Check_access { buffer; at } ->
      let b = buffer_of ctx block buffer in
      if is_flat at && Ssa_format.per_channel b.Ssa_buffer.format then
        fail ctx block (Flat_on_per_channel buffer)
  | Ssa_op.Load { buffer; decode; at }
  | Ssa_op.Load_in_bounds { buffer; decode; at } ->
      check_access ctx block buffer
        ~family:(Ssa_op.Decode.family decode)
        ~writes:false ~flat:(is_flat at)
  | Ssa_op.Store { buffer; encode; at; _ } ->
      check_access ctx block buffer
        ~family:(Ssa_op.Encode.family encode)
        ~writes:true ~flat:(is_flat at)
  | Ssa_op.Vec_load { buffer; decode; _ } ->
      check_access ctx block buffer
        ~family:(Ssa_op.Decode.family decode)
        ~writes:false ~flat:false
  | Ssa_op.Vec_store { buffer; encode; _ } ->
      check_access ctx block buffer
        ~family:(Ssa_op.Encode.family encode)
        ~writes:true ~flat:false
  | Ssa_op.Check_gather _ | Ssa_op.Check_local _ | Ssa_op.Check_scan _
  | Ssa_op.Const _ | Ssa_op.Convert _ | Ssa_op.Float_binary _
  | Ssa_op.Float_compare _ | Ssa_op.Float_fma _ | Ssa_op.Float_max _
  | Ssa_op.Float_to_i64 _ | Ssa_op.Float_unary _ | Ssa_op.I64_arith _
  | Ssa_op.I64_compare _ | Ssa_op.I64_div _ | Ssa_op.Index_add _
  | Ssa_op.Index_add_in_domain _ | Ssa_op.Index_ceil_div _
  | Ssa_op.Index_clamp_low _ | Ssa_op.Index_compare _ | Ssa_op.Index_floor_div _
  | Ssa_op.Index_max _ | Ssa_op.Index_min _ | Ssa_op.Index_of_i64 _
  | Ssa_op.Index_scale _ | Ssa_op.Index_scale_in_domain _ | Ssa_op.Lanewise _
  | Ssa_op.Local_alloc _ | Ssa_op.Local_read _ | Ssa_op.Local_write _
  | Ssa_op.Mark _ | Ssa_op.Mark_lanes _ | Ssa_op.Meter_charge
  | Ssa_op.Meter_release _ | Ssa_op.Meter_reserve _ | Ssa_op.Meter_reset
  | Ssa_op.Pool_better _ | Ssa_op.Pred_not _ | Ssa_op.Pred_or _
  | Ssa_op.Select _ | Ssa_op.Vec_extract _ | Ssa_op.Vec_insert _
  | Ssa_op.Vec_iota _ | Ssa_op.Vec_splat _ ->
      ()

let consume ctx block ~position ~live (e : Ssa_value.t) =
  use ctx block ~position e;
  if not (Ssa_value.equal e live) then
    fail ctx block (Effect_stale { used = e; live })

(* One operation: its operands in scope, its typing, and the effect chain. *)
let instr ctx block ~position ~live (i : Ssa_instr.t) =
  let op = i.Ssa_instr.op in
  List.iter (use ctx block ~position) (Ssa_op.operands op);
  check_buffers ctx block op;
  let values =
    match Ssa_typing.result_types op with
    | Ok ts -> ts
    | Error e -> fail ctx block (Typing e)
  in
  let expected =
    if Ssa_op.effectful op then values @ [ Ssa_type.Effect ] else values
  in
  let declared = types i.Ssa_instr.results in
  if
    not
      (List.length declared = List.length expected
      && List.for_all2 Ssa_type.equal declared expected)
  then fail ctx block (Result_types { declared; expected });
  match (Ssa_op.effectful op, i.Ssa_instr.token) with
  | true, Some e ->
      consume ctx block ~position ~live e;
      List.nth i.Ssa_instr.results (List.length i.Ssa_instr.results - 1)
  | false, None -> live
  | true, None | false, Some _ -> fail ctx block Effect_missing

(* The edge passes the chain to the target's effect parameter, if it has one,
   and every other argument to the parameters in order. *)
let edge ctx block ~position ~live (e : Ssa_cfg_edge.t) =
  match Ssa_cfg.find_block ctx.cfg e.target with
  | None -> fail ctx block (Target_unknown e.target)
  | Some target ->
      List.iter (use ctx block ~position) e.args;
      let expected = types target.Ssa_cfg_block.params
      and found = types e.args in
      if
        not
          (List.length expected = List.length found
          && List.for_all2 Ssa_type.equal expected found)
      then fail ctx block (Edge_arguments { expected; found });
      List.iter2
        (fun (p : Ssa_value.t) (a : Ssa_value.t) ->
          if is_effect p && not (Ssa_value.equal a live) then
            fail ctx block (Effect_stale { used = a; live }))
        target.Ssa_cfg_block.params e.args

let check (cfg : Ssa_cfg.t) =
  Err.Escape.with_escape @@ fun esc ->
  let entry = cfg.Ssa_cfg.entry in
  (match Ssa_cfg.find_block cfg entry with
  | None ->
      Err.Escape.throw esc
        (`Invalid_cfg { block = entry; problem = Entry_missing } : error)
  | Some _ -> ());
  let ctx0 =
    {
      esc;
      cfg;
      idom = Ssa_id.Block.Map.empty;
      preds = Ssa_cfg.predecessors cfg;
      defs = Ssa_id.Value.Map.empty;
    }
  in
  (* blocks once, and every target exists *)
  let seen = ref Ssa_id.Block.Set.empty in
  List.iter
    (fun (b : Ssa_cfg_block.t) ->
      if Ssa_id.Block.Set.mem b.id !seen then fail ctx0 b.id Block_defined_twice;
      seen := Ssa_id.Block.Set.add b.id !seen)
    cfg.Ssa_cfg.blocks;
  List.iter
    (fun (b : Ssa_cfg_block.t) ->
      List.iter
        (fun s ->
          if not (Ssa_id.Block.Set.mem s !seen) then
            fail ctx0 b.id (Target_unknown s))
        (Ssa_cfg_terminator.successors b.terminator))
    cfg.Ssa_cfg.blocks;
  (match Ssa_id.Block.Map.find_opt entry ctx0.preds with
  | Some (_ :: _) -> fail ctx0 entry Entry_has_predecessor
  | Some [] | None -> ());
  let rpo = Ssa_cfg.reverse_postorder cfg in
  List.iter
    (fun (b : Ssa_cfg_block.t) ->
      if not (List.exists (Ssa_id.Block.equal b.id) rpo) then
        fail ctx0 b.id Unreachable)
    cfg.Ssa_cfg.blocks;
  (* every definition, once *)
  let defs =
    List.fold_left
      (fun defs (b : Ssa_cfg_block.t) ->
        let define defs position (v : Ssa_value.t) =
          if Ssa_id.Value.Map.mem v.Ssa_value.id defs then
            fail ctx0 b.id (Definition_twice v.Ssa_value.id);
          Ssa_id.Value.Map.add v.Ssa_value.id
            (b.id, position, v.Ssa_value.ty)
            defs
        in
        let defs = List.fold_left (fun d v -> define d (-1) v) defs b.params in
        List.fold_left
          (fun (defs, position) (i : Ssa_instr.t) ->
            ( List.fold_left (fun d v -> define d position v) defs i.results,
              position + 1 ))
          (defs, 0) b.body
        |> fst)
      Ssa_id.Value.Map.empty cfg.Ssa_cfg.blocks
  in
  let ctx = { ctx0 with idom = Ssa_cfg.immediate_dominators cfg; defs } in
  (* the chain at the end of each block, in reverse postorder *)
  let ends = ref Ssa_id.Block.Map.empty in
  List.iter
    (fun id ->
      let b = Option.get (Ssa_cfg.find_block cfg id) in
      let params = b.Ssa_cfg_block.params in
      let live =
        match List.filter is_effect params with
        | [ e ] -> e
        | _ -> (
            match
              ( params,
                Option.value ~default:[]
                  (Ssa_id.Block.Map.find_opt id ctx.preds) )
            with
            | _, [ p ] when List.for_all (fun v -> not (is_effect v)) params
              -> (
                match Ssa_id.Block.Map.find_opt p !ends with
                | Some live -> live
                | None -> fail ctx id (Effect_parameters (types params)))
            | _ -> fail ctx id (Effect_parameters (types params)))
      in
      let live, position =
        List.fold_left
          (fun (live, position) i ->
            (instr ctx id ~position ~live i, position + 1))
          (live, 0) b.Ssa_cfg_block.body
      in
      (match b.Ssa_cfg_block.terminator with
      | Ssa_cfg_terminator.Branch { cond; then_; else_ } ->
          use ctx id ~position cond;
          if
            not
              (Ssa_type.equal cond.Ssa_value.ty (Ssa_type.Scalar Ssa_type.Pred))
          then fail ctx id (Branch_condition cond.Ssa_value.ty);
          edge ctx id ~position ~live then_;
          edge ctx id ~position ~live else_
      | Ssa_cfg_terminator.Jump e -> edge ctx id ~position ~live e
      | Ssa_cfg_terminator.Return e -> consume ctx id ~position ~live e);
      ends := Ssa_id.Block.Map.add id live !ends)
    rpo
