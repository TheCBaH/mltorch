module Class = struct
  type t = Data of Ssa_type.t | Erased
end

let classify (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Effect -> Class.Erased
  | ty -> Class.Data ty

module Move = struct
  type t = { destination : Ssa_value.t; source : Ssa_value.t }
end

let copies cfg (e : Ssa_cfg_edge.t) =
  match Ssa_cfg.find_block cfg e.target with
  | None -> invalid_arg "Ssa_cfg_handoff.copies: the target is not in the graph"
  | Some target ->
      List.concat
        (List.map2
           (fun (d : Ssa_value.t) (s : Ssa_value.t) ->
             match classify d with
             | Class.Erased -> []
             | Class.Data _ ->
                 if Ssa_value.equal d s then []
                 else [ { Move.destination = d; source = s } ])
           target.Ssa_cfg_block.params e.args)

(* Moves whose destination no pending move reads go first; when every pending
   destination is still read, the pending moves form cycles, and saving one
   destination's current value in a temporary and redirecting its readers
   there unblocks it. *)
let sequentialize ~fresh moves =
  let reads pending (v : Ssa_value.t) =
    List.exists (fun (m : Move.t) -> Ssa_value.equal m.source v) pending
  in
  let rec go pending out =
    match pending with
    | [] -> List.rev out
    | _ -> (
        match
          List.find_opt
            (fun (m : Move.t) ->
              not
                (reads (List.filter (fun m' -> m' != m) pending) m.destination))
            pending
        with
        | Some m -> go (List.filter (fun m' -> m' != m) pending) (m :: out)
        | None ->
            let m = List.hd pending in
            let saved = fresh m.destination.Ssa_value.ty in
            let save = { Move.destination = saved; source = m.destination } in
            let redirect (m' : Move.t) =
              if Ssa_value.equal m'.source m.destination then
                { m' with Move.source = saved }
              else m'
            in
            go (List.map redirect pending) (save :: out))
  in
  go moves []
