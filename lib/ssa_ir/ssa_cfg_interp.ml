module M = Ssa_interp.Machine

type error = [ Ssa_interp.failure | Ssa_cfg_verify.error ]

let pp_error fmt : [< error ] -> unit = function
  | `Invalid_cfg _ as e -> Ssa_cfg_verify.pp_error fmt e
  | #Ssa_interp.failure as e -> Ssa_interp.pp_error fmt e

let run ?counters ?fused (cfg : Ssa_cfg.t) ~memory =
  match Err.payload (Ssa_cfg_verify.check cfg) with
  | Error e -> Err.fail (e :> error)
  | Ok () ->
      let blocks = Hashtbl.create 16 in
      List.iter
        (fun (b : Ssa_cfg_block.t) -> Hashtbl.replace blocks (b.id :> int) b)
        cfg.Ssa_cfg.blocks;
      let block (id : Ssa_id.Block.t) = Hashtbl.find blocks (id :> int) in
      Err.map_error
        (fun (e : Ssa_interp.error) ->
          match e with
          | #Ssa_interp.failure as f -> (f :> error)
          | `Invalid_program _ ->
              invalid_arg "Ssa_cfg_interp: the machine verifies no program")
        (M.run ?counters ?fused ~buffers:cfg.Ssa_cfg.buffers
           ~scan_limits:cfg.Ssa_cfg.scan_limits
           ~next_value:cfg.Ssa_cfg.next_value ~memory (fun m ->
             (* every argument is read before any parameter is rebound *)
             let goto (e : Ssa_cfg_edge.t) =
               let target = block e.target in
               let values = List.map (M.read m) e.args in
               List.iter2 (M.write m) target.Ssa_cfg_block.params values;
               Some target
             in
             let current = ref (Some (block cfg.Ssa_cfg.entry)) in
             while !current <> None do
               match !current with
               | None -> ()
               | Some b -> (
                   List.iter (M.exec m) b.Ssa_cfg_block.body;
                   current :=
                     match b.Ssa_cfg_block.terminator with
                     | Ssa_cfg_terminator.Jump e -> goto e
                     | Ssa_cfg_terminator.Branch { cond; then_; else_ } ->
                         goto (if M.predicate m cond then then_ else else_)
                     | Ssa_cfg_terminator.Return _ -> None)
             done))
