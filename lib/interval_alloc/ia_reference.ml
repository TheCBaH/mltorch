(* Reference minimum-pool search: a bounded, exhaustive feasibility test for
   one ceiling, and the bisection over ceilings that turns its answers into
   proven bounds. Independent of the constructive strategies and the order
   search, so its answers can grade them.

   Feasibility is a search over orientations: two positive-size blocks whose
   lifetimes intersect must be stacked one above the other, and a set of such
   choices is a DAG whose longest paths are the lowest offsets it allows. Only
   a pair whose current longest-path ranges still overlap is branched on; when
   none do, those offsets are a placement. This is complete: a placement that
   fits the ceiling and honours the choices made so far orients the
   overlapping pair one way or the other, and adding that edge only raises
   offsets towards that placement's, so the branch that contains it is never
   pruned. A block's offset is the least aligned offset at or above its
   predecessors' ends; that stays complete, since an aligned placement at or
   above every one of those ends is at or above their aligned-up maximum.
   Every branch adds an edge no earlier choice implied, so the search is
   finite. A branch cut by a limit makes the answer [Unknown], never
   [Infeasible].

   No clock and no recursion: the depth-first search is a loop over an
   explicit stack, and work is counted in expanded states. *)

open Core.Storage_units
module Script = Ia_script

module Limits = struct
  type t = { max_states : int64; max_depth : int64 }
end

module Work = struct
  type t = { mutable states : int64; mutable max_depth : int64 }

  let create () = { states = 0L; max_depth = 0L }
  let states t = t.states
  let max_depth t = t.max_depth
end

module Cut = struct
  type t = Depth | States
end

(* ['w] is the witness a feasible answer carries. *)
module Answer = struct
  type 'w t = Feasible of 'w | Infeasible | Unknown of Cut.t
end

(* A feasible answer outside what the bisection has proven: below the lower
   bound, or above the ceiling it was asked about. *)
module Invalid_candidate = struct
  type t = { pool : Byte_size.t; lower : Byte_size.t; ceiling : Byte_size.t }
end

(* An undo record: a raised offset (with its end), or an edge pushed on a
   successor list. *)
type trail =
  | Edge of Script.Index.t
  | Offset of Script.Index.t * Byte_offset.t * Byte_offset.t

(* How much of the trail was recorded before a frame's branch. *)
module Mark =
  Core.Tagged_int.Make
    (struct
      let prefix = "m"
    end)
    ()

(* Which of a pair's two orientations a frame tries next: the larger block
   below, then above, then neither is left. *)
module Orientation = struct
  type t = Larger_below | Larger_above | Exhausted
end

(* One branching pair and which of its two orientations to try next. *)
type frame = {
  a : Script.Index.t;
  b : Script.Index.t;
  mutable next : Orientation.t;
  mark : Mark.t;
}

let feasible (limits : Limits.t) (work : Work.t) (script : _ Script.t) ~ceiling
    =
  let get = Script.get and set = Script.set in
  let sizes = script.Script.sizes and alignments = script.Script.alignments in
  let positive =
    List.filter
      (fun i -> not (Script.empty script i))
      (Array.to_list (Script.indices script))
  in
  if List.exists (fun i -> Byte_size.compare (get sizes i) ceiling > 0) positive
  then Ok Answer.Infeasible
  else
    let pairs =
      List.concat_map
        (fun a ->
          List.filter_map
            (fun b ->
              if a < b && Script.conflicts script a b then Some (a, b) else None)
            positive)
        positive
      |> Array.of_list
    in
    (* Every block starts at zero, which any alignment admits. *)
    let lb = Script.table script Byte_offset.zero
    and ends = Array.map Byte_offset.of_size sizes
    and succ = Script.table script [] in
    let trail = Stack.create () and queue = Queue.create () in
    let undo_to (mark : Mark.t) =
      while Stack.length trail > (mark :> int) do
        match Stack.pop trail with
        | Edge u -> set succ u (List.tl (get succ u))
        | Offset (x, v, e) ->
            set lb x v;
            set ends x e
      done
    in
    (* Raise [x]'s offset to the least aligned one at or above [v]; false
       when that cannot fit, or when it reaches [source], which closes a cycle
       through the new edge. An offset past [int64] does not fit either. *)
    let raise_to ~source x v =
      Byte_offset.compare v (get lb x) <= 0
      || x <> source
         &&
         match
           Result.bind
             (Byte_offset.align_up v (get alignments x))
             (fun o ->
               Result.map
                 (fun hi -> (o, hi))
                 (Byte_offset.advance o (get sizes x)))
         with
         | Ok (o, hi)
           when Byte_size.compare (Byte_offset.to_size hi) ceiling <= 0 ->
             Stack.push (Offset (x, get lb x, get ends x)) trail;
             set lb x o;
             set ends x hi;
             Queue.add x queue;
             true
         | Ok _ | Error _ -> false
    in
    (* [u] below [v], then longest paths to a fixpoint. *)
    let add_edge u v =
      Stack.push (Edge u) trail;
      set succ u (v :: get succ u);
      Queue.clear queue;
      let rec drain () =
        match Queue.take_opt queue with
        | None -> true
        | Some x ->
            List.for_all
              (fun y -> raise_to ~source:u y (get ends x))
              (get succ x)
            && drain ()
      in
      raise_to ~source:u v (get ends u) && drain ()
    in
    let overlap a b =
      Byte_offset.compare (get lb a) (get ends b) < 0
      && Byte_offset.compare (get lb b) (get ends a) < 0
    in
    (* The overlapping pair reaching highest: the one closest to breaking the
       ceiling, so a dead end shows up early. Ties keep pair order. *)
    let highest_overlap () =
      let best = ref None in
      Array.iter
        (fun (a, b) ->
          if overlap a b then
            let top = Byte_offset.max (get ends a) (get ends b) in
            match !best with
            | Some (t, _) when Byte_offset.compare t top >= 0 -> ()
            | _ -> best := Some (top, (a, b)))
        pairs;
      Option.map snd !best
    in
    let frames = Stack.create () and depth = ref 0L and depth_cut = ref false in
    let outcome = ref None and descend = ref true in
    while !outcome = None do
      if !descend then begin
        descend := false;
        match highest_overlap () with
        | None -> outcome := Some `Found
        | Some _ when Int64.compare !depth limits.Limits.max_depth >= 0 ->
            depth_cut := true
        | Some (a, b) ->
            Stack.push
              {
                a;
                b;
                next = Orientation.Larger_below;
                mark = Mark.of_int (Stack.length trail);
              }
              frames;
            depth := Int64.succ !depth
      end
      else
        match Stack.top_opt frames with
        | None -> outcome := Some `Exhausted
        | Some f ->
            undo_to f.mark;
            if f.next = Orientation.Exhausted then begin
              ignore (Stack.pop frames);
              depth := Int64.pred !depth
            end
            else if Int64.compare work.Work.states limits.Limits.max_states >= 0
            then outcome := Some `States
            else begin
              (* The larger block goes below first; a tie keeps allocation
                 order. *)
              let lo, hi =
                if Byte_size.compare (get sizes f.b) (get sizes f.a) > 0 then
                  (f.b, f.a)
                else (f.a, f.b)
              in
              let u, v =
                if f.next = Orientation.Larger_below then (lo, hi) else (hi, lo)
              in
              f.next <-
                (if f.next = Orientation.Larger_below then
                   Orientation.Larger_above
                 else Orientation.Exhausted);
              work.Work.states <- Int64.succ work.Work.states;
              work.Work.max_depth <- Int64.max work.Work.max_depth !depth;
              if add_edge u v then descend := true
            end
    done;
    match !outcome with
    | Some `Found ->
        let pool =
          List.fold_left
            (fun pool i ->
              Byte_size.max pool (Byte_offset.to_size (get ends i)))
            Byte_size.zero positive
        in
        let candidate =
          Ia_solution.Unsafe.make ~pool
            (List.map
               (fun i -> (Script.key script i, get lb i))
               (Array.to_list (Script.indices script)))
        in
        Result.map
          (fun w -> Answer.Feasible w)
          (Ia_check.check script candidate)
    | Some `States -> Ok (Answer.Unknown Cut.States)
    | Some `Exhausted | None ->
        Ok (if !depth_cut then Answer.Unknown Cut.Depth else Answer.Infeasible)

module Status = struct
  type t = Incomplete | Optimal_above_live_bound | Optimal_live_bound
end

module Stop = struct
  type t = Closed | Depth_limit | State_limit
end

module Bounds = struct
  type 'w t = {
    status : Status.t;
    stop : Stop.t;
    live_bound : Byte_size.t;
    initial_upper : Byte_size.t;
    lower : Byte_size.t;
    upper : Byte_size.t;
    incumbent : 'w;
    queries : int64;
  }
end

(* The bisection, over any feasibility oracle: [query c] answers whether a
   placement fits in [c], and [pool] is a feasible answer's actual length.
   The first query is the live bound itself; after it, the midpoint of what
   is still open below the incumbent. *)
let bisect ~live_bound ~upper:initial_upper ~incumbent ~pool ~query =
  let ( let* ) = Result.bind in
  let finish ~stop lower upper incumbent queries =
    let status =
      if Byte_size.compare lower upper < 0 then Status.Incomplete
      else if Byte_size.equal upper live_bound then Status.Optimal_live_bound
      else Status.Optimal_above_live_bound
    in
    Ok
      {
        Bounds.status;
        stop;
        live_bound;
        initial_upper;
        lower;
        upper;
        incumbent;
        queries;
      }
  in
  let rec go lower upper incumbent queries =
    if Byte_size.compare lower upper >= 0 then
      finish ~stop:Stop.Closed upper upper incumbent queries
    else
      (* [lower < upper], so [upper - 1] is a size and [ceiling < upper]. *)
      let ceiling =
        if Int64.equal queries 0L then lower
        else Byte_size.midpoint lower (Script.invariant (Byte_size.pred upper))
      in
      let queries = Int64.succ queries in
      let* answer = query ceiling in
      match answer with
      | Answer.Feasible w ->
          let p = pool w in
          if Byte_size.compare p lower < 0 || Byte_size.compare p ceiling > 0
          then
            Err.fail ~pos:__POS__
              (`Invalid_candidate { Invalid_candidate.pool = p; lower; ceiling })
          else go lower p w queries
      | Answer.Infeasible ->
          go (Script.invariant (Byte_size.succ ceiling)) upper incumbent queries
      | Answer.Unknown Cut.Depth ->
          finish ~stop:Stop.Depth_limit lower upper incumbent queries
      | Answer.Unknown Cut.States ->
          finish ~stop:Stop.State_limit lower upper incumbent queries
  in
  go live_bound initial_upper incumbent 0L

(* The incumbent's actual length: the highest occupied end, zero-size blocks
   at zero, placements in allocation order. Only for a checked solution, which
   places every key exactly once. *)
let normalize (script : _ Script.t) solution =
  let get = Script.get and set = Script.set in
  let offsets = Script.table script Byte_offset.zero in
  List.iter
    (fun (k, offset) ->
      match Script.find script k with
      | Some i -> if not (Script.empty script i) then set offsets i offset
      | None -> assert false)
    (Ia_solution.placements solution);
  (* Checked: every block ends inside the pool. *)
  let pool = ref Byte_size.zero in
  Array.iter
    (fun i ->
      let hi =
        Script.invariant
          (Byte_offset.advance (get offsets i) (Script.size script i))
      in
      pool := Byte_size.max !pool (Byte_offset.to_size hi))
    (Script.indices script);
  Ia_solution.Unsafe.make ~pool:!pool
    (List.map
       (fun i -> (Script.key script i, get offsets i))
       (Array.to_list (Script.indices script)))

let minimum limits script ~incumbent =
  let ( let* ) = Result.bind in
  let* live_bound = Script.lower_bound script in
  let* checked = Ia_check.check script incumbent in
  let* incumbent = Ia_check.check script (normalize script checked) in
  let work = Work.create () in
  let query ceiling = feasible limits work script ~ceiling in
  let* bounds =
    bisect ~live_bound
      ~upper:(Ia_solution.pool incumbent)
      ~incumbent ~pool:Ia_solution.pool ~query
  in
  Ok (bounds, work)
