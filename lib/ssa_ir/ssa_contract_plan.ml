module Unsupported = struct
  type t = {
    capabilities : Contraction.Select.Capabilities.t;
    desc : Contraction.t;
    engine : Contraction.Select.Engine.t;
  }
end

module Conflict = struct
  type t = {
    desc : Contraction.t;
    first : Contraction.Select.Engine.t;
    second : Contraction.Select.Engine.t;
  }
end

type error =
  [ `Contraction_engine_conflict of Conflict.t
  | `Unsupported_contraction_engine of Unsupported.t ]

let pp_error fmt = function
  | `Contraction_engine_conflict (c : Conflict.t) ->
      Fmt.pf fmt "%s has conflicting engines %s and %s"
        (Contraction.name c.desc)
        (Contraction.Select.Engine.name c.first)
        (Contraction.Select.Engine.name c.second)
  | `Unsupported_contraction_engine (u : Unsupported.t) ->
      Fmt.pf fmt "%s cannot realize %s (matrix=%b, vector=%b)"
        (Contraction.Select.Engine.name u.engine)
        (Contraction.name u.desc) u.capabilities.matrix u.capabilities.vector

let program ~capabilities p =
  Err.Escape.with_escape @@ fun esc ->
  let selected = Hashtbl.create 8 in
  let rule t = function
    | Ssa_stmt.Instr ({ Ssa_instr.op = Ssa_op.Contract c; _ } as i) ->
        let engine =
          match c.engine with
          | Some e -> e
          | None -> fst (Contraction.Select.choose capabilities c.desc)
        in
        if
          engine = Contraction.Select.Engine.Loops
          || not (Contraction.Select.supports capabilities c.desc engine)
        then
          Err.Escape.throw esc
            (`Unsupported_contraction_engine
               { Unsupported.capabilities; desc = c.desc; engine });
        let key = Contraction.name c.desc in
        (match Hashtbl.find_opt selected key with
        | Some first when first <> engine ->
            Err.Escape.throw esc
              (`Contraction_engine_conflict
                 { Conflict.desc = c.desc; first; second = engine })
        | Some _ | None -> Hashtbl.replace selected key engine);
        if c.engine <> Some engine then Ssa_rewrite.mark_changed t;
        [
          Ssa_stmt.Instr
            {
              i with
              Ssa_instr.op = Ssa_op.Contract { c with engine = Some engine };
            };
        ]
    | s -> [ s ]
  in
  let q, changed = Ssa_rewrite.program rule p in
  if changed then q else p

let choices p =
  let rec region (r : Ssa_region.t) =
    List.concat_map
      (function
        | Ssa_stmt.Instr
            { Ssa_instr.op = Ssa_op.Contract { desc; engine = Some e; _ }; _ }
          ->
            [ (desc, e) ]
        | Ssa_stmt.Instr _ -> []
        | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
            region body
        | Ssa_stmt.If { then_; else_; _ } -> region then_ @ region else_)
      r.body
  in
  region p.Ssa_program.entry
