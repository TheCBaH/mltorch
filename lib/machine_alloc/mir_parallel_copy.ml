(* Sequential moves for one simultaneous transfer between physical locations.
   Every source is read before any destination is written: a move whose
   destination no pending move still reads goes first; when only cycles
   remain, one destination's current value is saved to a scratch location
   that fits it, and the moves reading it read the scratch instead. Overlap,
   not name equality, decides whether a destination is still read, so partial
   views of one register are respected; a self-move needs the same bits. *)

open Machine_ir
module Loc = Mir_phys.Loc

type move = { dst : Loc.t; src : Loc.t; value : Mir_value.t }

let resolve ~(scratch : Mir_value.t -> Loc.t) (moves : move list) =
  let pending =
    ref (List.filter (fun m -> not (Loc.equal m.dst m.src)) moves)
  in
  let out = ref [] in
  let emit m = out := m :: !out in
  while !pending <> [] do
    let read_by_others m =
      List.exists (fun o -> o != m && Loc.overlap m.dst o.src) !pending
    in
    match List.find_opt (fun m -> not (read_by_others m)) !pending with
    | Some m ->
        emit m;
        pending := List.filter (fun o -> o != m) !pending
    | None -> (
        (* every pending destination is still read: a cycle *)
        match !pending with
        | m :: _ ->
            let readers =
              List.filter (fun o -> Loc.overlap m.dst o.src) !pending
            in
            (* saved as its readers read it: [m] may write a narrower or
               wider view of the register holding it *)
            let r = List.hd readers in
            let tmp = scratch r.value in
            emit { dst = tmp; src = r.src; value = r.value };
            pending :=
              List.map
                (fun o ->
                  if Loc.overlap m.dst o.src then { o with src = tmp } else o)
                !pending
        | [] -> ())
  done;
  List.rev !out
