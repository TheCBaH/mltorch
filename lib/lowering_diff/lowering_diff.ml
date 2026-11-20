type op = Delete of string | Equal of string | Insert of string
type t = { ops : op list; minimal : bool }

let lines s =
  if s = "" then []
  else
    let l = String.split_on_char '\n' s in
    (* a trailing newline ends the last line rather than starting another *)
    match List.rev l with
    | "" :: rest -> List.rev rest
    | _ -> l

(* Edit distance bound: the trace keeps one frontier per step, O(D^2) memory. *)
let max_edits = 1500

(* Myers' O(ND) greedy search over the edit graph. [trace.(d)] is the
   frontier [v] after step [d], used to walk back. Returns [None] past the
   bound. *)
let myers (a : string array) (b : string array) =
  let n = Array.length a and m = Array.length b in
  let max_d = min (n + m) max_edits in
  let offset = max_d + 1 in
  let v = Array.make ((2 * max_d) + 3) 0 in
  let trace = ref [] in
  let found = ref (-1) in
  (try
     for d = 0 to max_d do
       trace := Array.copy v :: !trace;
       let k = ref (-d) in
       while !k <= d do
         let kk = !k in
         let x =
           if kk = -d || (kk <> d && v.(offset + kk - 1) < v.(offset + kk + 1))
           then v.(offset + kk + 1)
           else v.(offset + kk - 1) + 1
         in
         let x = ref x in
         let y = ref (!x - kk) in
         while !x < n && !y < m && String.equal a.(!x) b.(!y) do
           incr x;
           incr y
         done;
         v.(offset + kk) <- !x;
         if !x >= n && !y >= m then (
           found := d;
           raise Exit);
         k := !k + 2
       done
     done
   with Exit -> ());
  if !found < 0 then None
  else
    (* [!trace] is newest first: index 0 is the frontier recorded at step
       [!found], i.e. the one {e before} that step's moves. Walk back from
       (n, m) choosing at each d the move that produced the frontier. *)
    let ops = ref [] in
    let x = ref n and y = ref m in
    let frontiers = Array.of_list (List.rev !trace) in
    for d = !found downto 0 do
      let vd = frontiers.(d) in
      let k = !x - !y in
      let prev_k =
        if k = -d || (k <> d && vd.(offset + k - 1) < vd.(offset + k + 1)) then
          k + 1
        else k - 1
      in
      let prev_x = vd.(offset + prev_k) in
      let prev_y = prev_x - prev_k in
      if d = 0 then
        (* only the initial diagonal slide remains *)
        while !x > 0 && !y > 0 do
          decr x;
          decr y;
          ops := Equal a.(!x) :: !ops
        done
      else (
        while !x > prev_x && !y > prev_y do
          decr x;
          decr y;
          ops := Equal a.(!x) :: !ops
        done;
        if !x = prev_x then (
          decr y;
          ops := Insert b.(!y) :: !ops)
        else (
          decr x;
          ops := Delete a.(!x) :: !ops))
    done;
    Some !ops

let diff old_text new_text =
  let a = Array.of_list (lines old_text)
  and b = Array.of_list (lines new_text) in
  match myers a b with
  | Some ops -> { ops; minimal = true }
  | None ->
      {
        ops =
          List.map (fun l -> Delete l) (Array.to_list a)
          @ List.map (fun l -> Insert l) (Array.to_list b);
        minimal = false;
      }

let identical t =
  List.for_all (function Equal _ -> true | Delete _ | Insert _ -> false) t.ops

module Hunk = struct
  type t = {
    old_start : int;
    old_count : int;
    new_start : int;
    new_count : int;
    ops : op list;
  }
end

let is_change = function Equal _ -> false | Delete _ | Insert _ -> true

let hunks ~context t =
  let ops = Array.of_list t.ops in
  let n = Array.length ops in
  (* a change at index [i] keeps [context] lines either side; overlapping or
     adjacent ranges merge *)
  let keep = Array.make n false in
  Array.iteri
    (fun i op ->
      if is_change op then
        for j = max 0 (i - context) to min (n - 1) (i + context) do
          keep.(j) <- true
        done)
    ops;
  let hunks = ref [] in
  let i = ref 0 in
  let old_line = ref 1 and new_line = ref 1 in
  while !i < n do
    if not keep.(!i) then (
      (match ops.(!i) with
      | Equal _ ->
          incr old_line;
          incr new_line
      | Delete _ -> incr old_line
      | Insert _ -> incr new_line);
      incr i)
    else begin
      let start = !i in
      let os = !old_line and ns = !new_line in
      let oc = ref 0 and nc = ref 0 in
      while !i < n && keep.(!i) do
        (match ops.(!i) with
        | Equal _ ->
            incr oc;
            incr nc
        | Delete _ -> incr oc
        | Insert _ -> incr nc);
        incr i
      done;
      old_line := !old_line + !oc;
      new_line := !new_line + !nc;
      hunks :=
        {
          Hunk.old_start = (if !oc = 0 then os - 1 else os);
          old_count = !oc;
          new_start = (if !nc = 0 then ns - 1 else ns);
          new_count = !nc;
          ops = Array.to_list (Array.sub ops start (!i - start));
        }
        :: !hunks
    end
  done;
  List.rev !hunks

let unified ~context ~old_label ~new_label t =
  match hunks ~context t with
  | [] -> ""
  | hs ->
      let b = Buffer.create 256 in
      Printf.bprintf b "--- %s\n+++ %s\n" old_label new_label;
      List.iter
        (fun (h : Hunk.t) ->
          Printf.bprintf b "@@ -%d,%d +%d,%d @@\n" h.old_start h.old_count
            h.new_start h.new_count;
          List.iter
            (function
              | Equal l -> Printf.bprintf b " %s\n" l
              | Delete l -> Printf.bprintf b "-%s\n" l
              | Insert l -> Printf.bprintf b "+%s\n" l)
            h.ops)
        hs;
      Buffer.contents b
