module P = Ssa_pp.Parts

let to_string (cfg : Ssa_cfg.t) =
  let st = P.create () in
  List.iter (P.buffer st) cfg.Ssa_cfg.buffers;
  let order = Ssa_cfg.reverse_postorder cfg in
  let label id =
    let rec find i = function
      | [] -> "bb?"
      | x :: rest ->
          if Ssa_id.Block.equal x id then Fmt.str "bb%d" i
          else find (i + 1) rest
    in
    find 0 order
  in
  let edge st (e : Ssa_cfg_edge.t) =
    Fmt.str "%s(%s)" (label e.target) (P.uses st e.args)
  in
  List.iter
    (fun id ->
      match Ssa_cfg.find_block cfg id with
      | None -> ()
      | Some b -> (
          P.line st 0 "%s(%s):" (label id) (P.defs st b.Ssa_cfg_block.params);
          List.iter (P.instr st 1) b.Ssa_cfg_block.body;
          match b.Ssa_cfg_block.terminator with
          | Ssa_cfg_terminator.Branch { cond; then_; else_ } ->
              P.line st 1 "branch %s, %s, %s" (P.name st cond) (edge st then_)
                (edge st else_)
          | Ssa_cfg_terminator.Jump e -> P.line st 1 "jump %s" (edge st e)
          | Ssa_cfg_terminator.Return e -> P.line st 1 "return %s" (P.name st e)
          ))
    order;
  P.contents st

let pp fmt cfg = Fmt.string fmt (to_string cfg)
