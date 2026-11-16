module B = Ssa_builder
open Ssa_expand_util

(* The template makes three placeholder constants, outside the index range any
   expansion constant uses, standing for the bases; the splice substitutes the
   call's own values for them, and the placeholders' definitions are dropped. *)
let placeholder n = Int64.sub Ssa_const.index_max (Int64.of_int n)

let program (p : Ssa_program.t) =
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.Instr
        {
          Ssa_instr.op = Ssa_op.View_call { desc; a; b; c };
          token = Some token;
          results = [ after ];
          _;
        }
      when Ssa_expand_contract.expandable desc ->
        let holders = ref [] in
        let bases bld =
          let h n = B.index bld (placeholder n) in
          let ha = h 0 and hb = h 1 and hc = h 2 in
          holders :=
            [ (ha :> Ssa_value.t); (hb :> Ssa_value.t); (hc :> Ssa_value.t) ];
          (ha, hb, hc)
        in
        let ids =
          [
            a.Ssa_view_call.View.buffer;
            b.Ssa_view_call.View.buffer;
            c.Ssa_view_call.View.buffer;
          ]
        in
        let template =
          Ssa_expand_contract.expansion ~bases desc
            ~a:a.Ssa_view_call.View.buffer ~b:b.Ssa_view_call.View.buffer
            ~c:c.Ssa_view_call.View.buffer ~buffers:(declared p ids)
        in
        let entry = template.Ssa_program.entry in
        let is_holder (i : Ssa_instr.t) =
          List.exists
            (fun (h : Ssa_value.t) ->
              List.exists
                (fun (r : Ssa_value.t) ->
                  Ssa_id.Value.equal r.Ssa_value.id h.Ssa_value.id)
                i.Ssa_instr.results)
            !holders
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
              :: List.map2
                   (fun h (v : Ssa_view_call.View.t) ->
                     (h, v.Ssa_view_call.View.base))
                   !holders [ a; b; c ])
        in
        let stmts = Ssa_clone.stmts cl body in
        Ssa_rewrite.alias t ~from:after
          ~to_:(Ssa_clone.value cl (List.hd entry.Ssa_region.yields));
        stmts
    | _ -> [ s ]
  in
  fst (Ssa_rewrite.program rule p)
