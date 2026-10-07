(* The structural verification every stage shares: unique ids and single
   definitions, reachability, dominance of every use, edge and block-parameter
   signatures, the order chain, object and helper declarations. A stage
   supplies its opcode typing and terminator rules; its own semantic checks run
   over the analysis this returns. *)

module D = Mir_diagnostic
module P = Mir_diagnostic.Problem

type context = {
  signature : Mir_op.Callee.t -> Mir_typing.Signature.t option;
  view : Mir_id.View.t -> Mir_view.t option;
}

module type STAGE = sig
  type op
  type term

  val stage : D.Stage.t
  val operands : op -> Mir_value.t list
  val ordered : op -> bool
  val typing : context -> op -> (Mir_type.t list, P.t) result
  val edges : term -> Mir_edge.t list

  val term_values : term -> Mir_value.t list
  (** non-edge value operands: a condition, return values, a payload *)

  val term_order : term -> Mir_value.t option
  (** the order state a non-edge exit consumes *)

  val check_term :
    context -> results:Mir_type.t list -> term -> (unit, P.t) result
end

(* Where a value is defined. *)
module Def = struct
  type 'op t =
    | Order_out of 'op Mir_instr.t
    | Order_param of Mir_id.Block.t
    | Param of Mir_id.Block.t
    | Result of 'op Mir_instr.t * int  (** the instruction and its position *)
end

module Analysis = struct
  type 'op t = {
    graph : Mir_graph.t;
    defs : ('op Def.t * Mir_id.Block.t * int) Mir_id.Value.Map.t;
        (** definition, block and position (-1 for parameters) *)
  }
end

(* A rejection leaves the walk through its escape frame, carrying the first
   problem found. *)
let make_reject esc stage ?func ?block ?instr problem =
  Err.Escape.throw esc { D.stage; func; block; instr; problem }

(* The program's declarations: regions, views and helpers. *)
let objects esc stage (regions : Mir_region.t list) (views : Mir_view.t list)
    (helpers : Mir_helper.t list) =
  let reject p = make_reject esc stage p in
  let seen_r = ref Mir_id.Region.Set.empty in
  List.iter
    (fun (r : Mir_region.t) ->
      if Mir_id.Region.Set.mem r.Mir_region.id !seen_r then
        reject (P.Duplicate_region r.Mir_region.id);
      seen_r := Mir_id.Region.Set.add r.Mir_region.id !seen_r;
      let ok =
        Int64.compare r.Mir_region.size 0L >= 0
        && Mir_layout.is_power_of_two r.Mir_region.align
        && Int64.compare r.Mir_region.align Mir_layout.max_align <= 0
        &&
        match r.Mir_region.init with
        | Mir_region.Constant s ->
            Int64.equal (Int64.of_int (String.length s)) r.Mir_region.size
        | Mir_region.Bound | Mir_region.Uninitialized -> true
      in
      if not ok then reject (P.Bad_region r.Mir_region.id))
    regions;
  let seen_v = ref Mir_id.View.Set.empty in
  List.iter
    (fun (v : Mir_view.t) ->
      if Mir_id.View.Set.mem v.Mir_view.id !seen_v then
        reject (P.Duplicate_view v.Mir_view.id);
      seen_v := Mir_id.View.Set.add v.Mir_view.id !seen_v;
      let region =
        List.find_opt
          (fun (r : Mir_region.t) ->
            Mir_id.Region.equal r.Mir_region.id v.Mir_view.region)
          regions
      in
      let ok =
        match region with
        | None -> false
        | Some r -> (
            Int64.compare v.Mir_view.offset 0L >= 0
            && Int64.compare v.Mir_view.size 0L >= 0
            && (match Mir_layout.add v.Mir_view.offset v.Mir_view.size with
              | Some e -> Int64.compare e r.Mir_region.size <= 0
              | None -> false)
            &&
            match r.Mir_region.init with
            | Mir_region.Constant _ -> not (Mir_view.writable v.Mir_view.perm)
            | Mir_region.Bound | Mir_region.Uninitialized -> true)
      in
      if not ok then reject (P.Bad_view v.Mir_view.id))
    views;
  let seen_h = ref Mir_id.Helper.Set.empty in
  List.iter
    (fun (h : Mir_helper.t) ->
      if Mir_id.Helper.Set.mem h.Mir_helper.id !seen_h then
        reject (P.Duplicate_helper h.Mir_helper.id);
      seen_h := Mir_id.Helper.Set.add h.Mir_helper.id !seen_h;
      if
        not
          (List.for_all Mir_type.has_storage
             (h.Mir_helper.params @ h.Mir_helper.results)
          && h.Mir_helper.name <> "")
      then reject (P.Bad_helper h.Mir_helper.id))
    helpers

module Make (S : STAGE) = struct
  let func esc cx ~(seen_instrs : Mir_id.Instr.Set.t ref)
      (f : (S.op, S.term) Mir_func.t) =
    let fid = f.Mir_func.id in
    let reject ?block ?instr p =
      make_reject esc S.stage ~func:fid ?block ?instr p
    in
    (* blocks: unique, entry present *)
    let blocks =
      List.fold_left
        (fun m (b : (S.op, S.term) Mir_block.t) ->
          if Mir_id.Block.Map.mem b.Mir_block.id m then
            reject (P.Duplicate_block b.Mir_block.id);
          Mir_id.Block.Map.add b.Mir_block.id b m)
        Mir_id.Block.Map.empty f.Mir_func.blocks
    in
    if not (Mir_id.Block.Map.mem f.Mir_func.entry blocks) then
      reject (P.Missing_block f.Mir_func.entry);
    let edges_of (b : (S.op, S.term) Mir_block.t) =
      S.edges b.Mir_block.terminator
    in
    List.iter
      (fun (b : (S.op, S.term) Mir_block.t) ->
        List.iter
          (fun (e : Mir_edge.t) ->
            if not (Mir_id.Block.Map.mem e.Mir_edge.target blocks) then
              reject ~block:b.Mir_block.id (P.Missing_block e.Mir_edge.target);
            if Mir_id.Block.equal e.Mir_edge.target f.Mir_func.entry then
              reject ~block:b.Mir_block.id
                (P.Entry_has_predecessor f.Mir_func.entry))
          (edges_of b))
      f.Mir_func.blocks;
    let graph =
      Mir_graph.make ~entry:f.Mir_func.entry
        (List.map
           (fun (b : (S.op, S.term) Mir_block.t) ->
             ( b.Mir_block.id,
               List.map (fun (e : Mir_edge.t) -> e.Mir_edge.target) (edges_of b)
             ))
           f.Mir_func.blocks)
    in
    List.iter
      (fun (b : (S.op, S.term) Mir_block.t) ->
        if not (Mir_graph.reachable graph b.Mir_block.id) then
          reject (P.Unreachable_block b.Mir_block.id))
      f.Mir_func.blocks;
    (* definitions: each value once *)
    let defs = ref Mir_id.Value.Map.empty in
    let define ?instr block pos (v : Mir_value.t) d =
      if Mir_id.Value.Map.mem v.Mir_value.id !defs then
        reject ~block ?instr (P.Duplicate_value v.Mir_value.id);
      defs := Mir_id.Value.Map.add v.Mir_value.id (d, block, pos) !defs
    in
    List.iter
      (fun (b : (S.op, S.term) Mir_block.t) ->
        let bid = b.Mir_block.id in
        List.iter
          (fun (p : Mir_value.t) ->
            if not (Mir_type.has_storage p.Mir_value.ty) then
              reject ~block:bid (P.Value_type p.Mir_value.id);
            define bid (-1) p (Def.Param bid))
          b.Mir_block.params;
        if not (Mir_type.equal b.Mir_block.order.Mir_value.ty Mir_type.Order)
        then reject ~block:bid (P.Value_type b.Mir_block.order.Mir_value.id);
        define bid (-1) b.Mir_block.order (Def.Order_param bid);
        List.iteri
          (fun pos (i : S.op Mir_instr.t) ->
            let iid = i.Mir_instr.id in
            if Mir_id.Instr.Set.mem iid !seen_instrs then
              reject ~block:bid ~instr:iid (P.Duplicate_instr iid);
            seen_instrs := Mir_id.Instr.Set.add iid !seen_instrs;
            List.iteri
              (fun k r -> define ~instr:iid bid pos r (Def.Result (i, k)))
              i.Mir_instr.results;
            match i.Mir_instr.order with
            | Some { Mir_order.output; _ } ->
                define ~instr:iid bid pos output (Def.Order_out i)
            | None -> ())
          b.Mir_block.body)
      f.Mir_func.blocks;
    let defs = !defs in
    (* a use in [block] at [pos] (the body length for the terminator) *)
    let use ?instr block pos (v : Mir_value.t) =
      match Mir_id.Value.Map.find_opt v.Mir_value.id defs with
      | None -> reject ~block ?instr (P.Undefined_value v.Mir_value.id)
      | Some (_, db, dpos) ->
          let ok =
            if Mir_id.Block.equal db block then dpos < pos
            else Mir_graph.dominates graph db block
          in
          if not ok then reject ~block ?instr (P.Not_dominated v.Mir_value.id)
    in
    let def_type (v : Mir_value.t) =
      match Mir_id.Value.Map.find_opt v.Mir_value.id defs with
      | Some (Def.Param b, _, _) -> (
          match Mir_id.Block.Map.find_opt b blocks with
          | Some blk ->
              List.find_map
                (fun (p : Mir_value.t) ->
                  if Mir_value.equal p v then Some p.Mir_value.ty else None)
                blk.Mir_block.params
          | None -> None)
      | Some (Def.Result (i, k), _, _) ->
          Option.map
            (fun (r : Mir_value.t) -> r.Mir_value.ty)
            (List.nth_opt i.Mir_instr.results k)
      | Some ((Def.Order_param _ | Def.Order_out _), _, _) ->
          Some Mir_type.Order
      | None -> None
    in
    (* a use names the type its definition gave it *)
    let check_use_type ?instr block (v : Mir_value.t) =
      match def_type v with
      | Some ty when Mir_type.equal ty v.Mir_value.ty -> ()
      | Some _ -> reject ~block ?instr (P.Value_type v.Mir_value.id)
      | None -> ()
    in
    List.iter
      (fun (b : (S.op, S.term) Mir_block.t) ->
        let bid = b.Mir_block.id in
        let current = ref b.Mir_block.order in
        List.iteri
          (fun pos (i : S.op Mir_instr.t) ->
            let instr = i.Mir_instr.id in
            let reject p = reject ~block:bid ~instr p in
            List.iter
              (fun v ->
                use ~instr bid pos v;
                check_use_type ~instr bid v;
                if Mir_type.equal v.Mir_value.ty Mir_type.Order then
                  reject (P.Value_type v.Mir_value.id))
              (S.operands i.Mir_instr.op);
            (match S.typing cx i.Mir_instr.op with
            | Error p -> reject p
            | Ok tys ->
                if
                  List.length tys <> List.length i.Mir_instr.results
                  || not
                       (List.for_all2
                          (fun t (r : Mir_value.t) ->
                            Mir_type.equal t r.Mir_value.ty)
                          tys i.Mir_instr.results)
                then reject P.Result_mismatch);
            match (S.ordered i.Mir_instr.op, i.Mir_instr.order) with
            | true, Some { Mir_order.input; output } ->
                use ~instr bid pos input;
                if not (Mir_value.equal input !current) then
                  reject (P.Stale_order input.Mir_value.id);
                if not (Mir_type.equal output.Mir_value.ty Mir_type.Order) then
                  reject (P.Value_type output.Mir_value.id);
                current := output
            | true, None -> reject P.Order_expected
            | false, Some _ -> reject P.Order_unexpected
            | false, None -> ())
          b.Mir_block.body;
        let pos = List.length b.Mir_block.body in
        let term = b.Mir_block.terminator in
        let reject p = reject ~block:bid p in
        List.iter
          (fun v ->
            use bid pos v;
            check_use_type bid v;
            if Mir_type.equal v.Mir_value.ty Mir_type.Order then
              reject (P.Value_type v.Mir_value.id))
          (S.term_values term);
        (match S.term_order term with
        | Some o ->
            use bid pos o;
            if not (Mir_value.equal o !current) then
              reject (P.Stale_order o.Mir_value.id)
        | None -> ());
        List.iter
          (fun (e : Mir_edge.t) ->
            let target = Mir_id.Block.Map.find e.Mir_edge.target blocks in
            let params = target.Mir_block.params in
            let expected = List.length params
            and found = List.length e.Mir_edge.args in
            if expected <> found then
              reject
                (P.Edge_arity { target = e.Mir_edge.target; expected; found });
            List.iteri
              (fun position ((a : Mir_value.t), (p : Mir_value.t)) ->
                use bid pos a;
                check_use_type bid a;
                if not (Mir_type.equal a.Mir_value.ty p.Mir_value.ty) then
                  reject (P.Edge_type { target = e.Mir_edge.target; position }))
              (List.combine e.Mir_edge.args params);
            use bid pos e.Mir_edge.order;
            if not (Mir_value.equal e.Mir_edge.order !current) then
              reject (P.Stale_order e.Mir_edge.order.Mir_value.id))
          (S.edges term);
        match S.check_term cx ~results:f.Mir_func.results term with
        | Ok () -> ()
        | Error p -> reject p)
      f.Mir_func.blocks;
    { Analysis.graph; defs }

  (* Every function's structure; [Error] carries the first rejection. *)
  let program (p : (S.op, S.term) Mir_program.t) =
    Err.Escape.with_escape @@ fun esc ->
    let reject ?func p = make_reject esc S.stage ?func p in
    objects esc S.stage p.Mir_program.regions p.Mir_program.views
      p.Mir_program.helpers;
    let seen_f = ref Mir_id.Func.Set.empty in
    List.iter
      (fun (f : (S.op, S.term) Mir_func.t) ->
        if Mir_id.Func.Set.mem f.Mir_func.id !seen_f then
          reject (P.Duplicate_func f.Mir_func.id);
        seen_f := Mir_id.Func.Set.add f.Mir_func.id !seen_f)
      p.Mir_program.funcs;
    if Option.is_none (Mir_program.find_func p p.Mir_program.main) then
      reject (P.Missing_main p.Mir_program.main);
    let signature = function
      | Mir_op.Callee.Func id ->
          Option.map
            (fun (f : (S.op, S.term) Mir_func.t) ->
              {
                Mir_typing.Signature.params =
                  List.map
                    (fun (v : Mir_value.t) -> v.Mir_value.ty)
                    (Mir_func.params f);
                results = f.Mir_func.results;
              })
            (Mir_program.find_func p id)
      | Mir_op.Callee.Helper id ->
          Option.map
            (fun (h : Mir_helper.t) ->
              {
                Mir_typing.Signature.params = h.Mir_helper.params;
                results = h.Mir_helper.results;
              })
            (Mir_program.find_helper p id)
    in
    let cx = { signature; view = Mir_program.find_view p } in
    let analyses =
      List.map
        (fun (f : (S.op, S.term) Mir_func.t) ->
          if not (List.for_all Mir_type.has_storage f.Mir_func.results) then
            reject ~func:f.Mir_func.id P.Return_mismatch;
          (* instruction ids are unique within a function *)
          (f, func esc cx ~seen_instrs:(ref Mir_id.Instr.Set.empty) f))
        p.Mir_program.funcs
    in
    (cx, analyses)
end
