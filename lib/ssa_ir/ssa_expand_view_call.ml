module B = Ssa_builder
open Ssa_expand_util

(* The template makes one placeholder constant per base, outside the index range
   any expansion constant uses, standing for the base; the splice substitutes
   the call's own values for them, and the placeholders' definitions are
   dropped. *)
let placeholder n = Int64.sub Ssa_const.index_max (Int64.of_int n)

let program (p : Ssa_program.t) =
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr
        {
          Ssa_instr.op = Ssa_op.View_call ({ desc; a; b; c } as call);
          token = Some token;
          results = [ after ];
          _;
        }
      when Ssa_expand_contract.expandable desc
           && Ssa_view_call.rows_expected call ->
        let holders = ref [] in
        let views bld =
          let next = ref 0 in
          let holder () =
            let h = B.index bld (placeholder !next) in
            incr next;
            holders := (h :> Ssa_value.t) :: !holders;
            h
          in
          let view (v : Ssa_view_call.View.t) =
            match v.Ssa_view_call.View.at with
            | Ssa_view_call.At.Base _ -> Ssa_expand_contract.Base (holder ())
            | Ssa_view_call.At.Rows rs ->
                Ssa_expand_contract.Rows
                  (Array.of_list (List.map (fun _ -> holder ()) rs))
          in
          let va = view a in
          let vb = view b in
          let vc = view c in
          (va, vb, vc)
        in
        let buffer (v : Ssa_view_call.View.t) = v.Ssa_view_call.View.buffer in
        let template =
          Ssa_expand_contract.expansion ~views desc ~a:(buffer a) ~b:(buffer b)
            ~c:(buffer c)
            ~buffers:(declared p [ buffer a; buffer b; buffer c ])
        in
        let holders = List.rev !holders in
        let entry = template.Ssa_program.entry in
        let is_holder (i : Ssa_instr.t) =
          List.exists
            (fun (h : Ssa_value.t) ->
              List.exists
                (fun (r : Ssa_value.t) ->
                  Ssa_id.Value.equal r.Ssa_value.id h.Ssa_value.id)
                i.Ssa_instr.results)
            holders
        in
        let body =
          List.filter
            (function Ssa_stmt.Instr i -> not (is_holder i) | _ -> true)
            entry.Ssa_region.body
        in
        let cl =
          Ssa_clone.create t
            ~subst:
              ((List.hd entry.Ssa_region.params, token)
              :: List.combine holders (Ssa_view_call.operands call))
        in
        let stmts = Ssa_clone.stmts cl body in
        Ssa_rewrite.alias t ~from:after
          ~to_:(Ssa_clone.value cl (List.hd entry.Ssa_region.yields));
        stmts
    | _ -> [ s ]
  in
  fst (Ssa_rewrite.program rule p)
