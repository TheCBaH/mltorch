(* A [For] whose bounds are both literal [Const]s exactly one apart runs its
   body exactly once, unconditionally: no [Loop_range] lookup is needed, since
   the raw lowering's own axis nest ([Loop_lower.nest_with]) already builds
   every non-window loop's bounds as [Const 0]/[Const (extent a)], and the
   window loops (conv/pool/pad) that don't are deliberately left alone here --
   their bounds are not literal constants, so they never match.

   Substitution reuses [Loop_index_map]: a loop variable is bound exactly once
   ([Loop_var.Next.first] is a monotone supply, never reused), so there is no
   shadowing to worry about -- replacing every occurrence of [Var v] wherever
   it appears in the body, including nested loops that bind their own,
   different, variable, is always correct.

   The extent is taken in [int64]: under js_of_ocaml [h - l] is 32-bit, and
   [l = 2^31 - 1, h = -2^31] (an empty loop) wraps to exactly [1]. *)
let rec rewrite ~replacement (stmts : Loop_stmt.t list) : Loop_stmt.t list =
  List.concat_map
    (fun (stmt : Loop_stmt.t) ->
      match stmt with
      | Loop_stmt.For
          { var; lo = Loop_index.Const l; hi = Loop_index.Const h; body }
        when Int64.(equal (sub (of_int h) (of_int l)) 1L) ->
          let r = replacement (Loop_index.Const l) in
          Loop_index_map.stmts
            ~f:(function
              | Loop_index.Var v' when Loop_var.equal var v' -> r | idx -> idx)
            (rewrite ~replacement body)
      | Loop_stmt.For { var; lo; hi; body } ->
          [ Loop_stmt.For { var; lo; hi; body = rewrite ~replacement body } ]
      | Loop_stmt.If (p, a, b) ->
          [ Loop_stmt.If (p, rewrite ~replacement a, rewrite ~replacement b) ]
      | s -> [ s ])
    stmts

let with_substitute replacement program =
  {
    program with
    Loop_program.body = rewrite ~replacement program.Loop_program.body;
  }

let run = with_substitute Fun.id
