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
   pruned. Every branch adds an edge no earlier choice implied, so the search
   is finite. A branch cut by a limit makes the answer [Unknown], never
   [Infeasible].

   No clock and no recursion: the depth-first search is a loop over an
   explicit stack, and work is counted in expanded states. *)

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
  type t = { pool : int64; lower : int64; ceiling : int64 }
end

(* An undo record: a raised offset, or an edge pushed on a successor list. *)
type trail = Edge of int | Offset of int * int64

(* One branching pair and which of its two orientations to try next. *)
type frame = { a : int; b : int; mutable next : int; mark : int }

let feasible (limits : Limits.t) (work : Work.t) (script : _ Script.t) ~ceiling
    =
  let n = Script.blocks script in
  let sizes = script.Script.sizes in
  let positive = List.filter (fun i -> sizes.(i) <> 0L) (List.init n Fun.id) in
  if List.exists (fun i -> Int64.compare sizes.(i) ceiling > 0) positive then
    Ok Answer.Infeasible
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
    let lb = Array.make n 0L and succ = Array.make n [] in
    let trail = Stack.create () and queue = Queue.create () in
    let undo_to mark =
      while Stack.length trail > mark do
        match Stack.pop trail with
        | Edge u -> succ.(u) <- List.tl succ.(u)
        | Offset (x, v) -> lb.(x) <- v
      done
    in
    (* Raise [x]'s offset to at least [v]; false when that cannot fit, or when
       it reaches [source], which closes a cycle through the new edge. *)
    let raise_to ~source x v =
      if Int64.compare v lb.(x) <= 0 then true
      else if x = source then false
      else
        match Script.add_checked v sizes.(x) with
        | Some hi when Int64.compare hi ceiling <= 0 ->
            Stack.push (Offset (x, lb.(x))) trail;
            lb.(x) <- v;
            Queue.add x queue;
            true
        | Some _ | None -> false
    in
    (* [u] below [v], then longest paths to a fixpoint. *)
    let add_edge u v =
      Stack.push (Edge u) trail;
      succ.(u) <- v :: succ.(u);
      Queue.clear queue;
      let rec drain () =
        match Queue.take_opt queue with
        | None -> true
        | Some x ->
            let hi = Int64.add lb.(x) sizes.(x) in
            if List.for_all (fun y -> raise_to ~source:u y hi) succ.(x) then
              drain ()
            else false
      in
      raise_to ~source:u v (Int64.add lb.(u) sizes.(u)) && drain ()
    in
    let overlap a b =
      Int64.compare lb.(a) (Int64.add lb.(b) sizes.(b)) < 0
      && Int64.compare lb.(b) (Int64.add lb.(a) sizes.(a)) < 0
    in
    (* The overlapping pair reaching highest: the one closest to breaking the
       ceiling, so a dead end shows up early. Ties keep pair order. *)
    let highest_overlap () =
      let best = ref None in
      Array.iter
        (fun (a, b) ->
          if overlap a b then
            let top =
              Int64.max
                (Int64.add lb.(a) sizes.(a))
                (Int64.add lb.(b) sizes.(b))
            in
            match !best with
            | Some (t, _) when Int64.compare t top >= 0 -> ()
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
            Stack.push { a; b; next = 0; mark = Stack.length trail } frames;
            depth := Int64.succ !depth
      end
      else
        match Stack.top_opt frames with
        | None -> outcome := Some `Exhausted
        | Some f ->
            undo_to f.mark;
            if f.next >= 2 then begin
              ignore (Stack.pop frames);
              depth := Int64.pred !depth
            end
            else if Int64.compare work.Work.states limits.Limits.max_states >= 0
            then outcome := Some `States
            else begin
              (* The larger block goes below first; a tie keeps allocation
                 order. *)
              let lo, hi =
                if Int64.compare sizes.(f.b) sizes.(f.a) > 0 then (f.b, f.a)
                else (f.a, f.b)
              in
              let u, v = if f.next = 0 then (lo, hi) else (hi, lo) in
              f.next <- f.next + 1;
              work.Work.states <- Int64.succ work.Work.states;
              work.Work.max_depth <- Int64.max work.Work.max_depth !depth;
              if add_edge u v then descend := true
            end
    done;
    match !outcome with
    | Some `Found ->
        let pool =
          List.fold_left
            (fun pool i -> Int64.max pool (Int64.add lb.(i) sizes.(i)))
            0L positive
        in
        let candidate =
          Ia_solution.Unsafe.make ~pool
            (List.init n (fun i -> (script.Script.keys.(i), lb.(i))))
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
    live_bound : int64;
    initial_upper : int64;
    lower : int64;
    upper : int64;
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
      if Int64.compare lower upper < 0 then Status.Incomplete
      else if Int64.equal upper live_bound then Status.Optimal_live_bound
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
    if Int64.compare lower upper >= 0 then
      finish ~stop:Stop.Closed upper upper incumbent queries
    else
      let ceiling =
        if Int64.equal queries 0L then lower
        else Int64.add lower (Int64.div (Int64.sub (Int64.pred upper) lower) 2L)
      in
      let queries = Int64.succ queries in
      let* answer = query ceiling in
      match answer with
      | Answer.Feasible w ->
          let p = pool w in
          if Int64.compare p lower < 0 || Int64.compare p ceiling > 0 then
            Err.fail ~pos:__POS__
              (`Invalid_candidate { Invalid_candidate.pool = p; lower; ceiling })
          else go lower p w queries
      | Answer.Infeasible -> go (Int64.succ ceiling) upper incumbent queries
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
  let n = Script.blocks script in
  let sizes = script.Script.sizes and keys = script.Script.keys in
  let offsets = Array.make n 0L in
  List.iter
    (fun (k, offset) ->
      match Script.find_index script.Script.equal keys n k with
      | Some i -> if sizes.(i) <> 0L then offsets.(i) <- offset
      | None -> assert false)
    (Ia_solution.placements solution);
  let pool = ref 0L in
  Array.iteri
    (fun i offset -> pool := Int64.max !pool (Int64.add offset sizes.(i)))
    offsets;
  Ia_solution.Unsafe.make ~pool:!pool
    (List.init n (fun i -> (keys.(i), offsets.(i))))

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
