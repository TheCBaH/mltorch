(* A local PCG-like generator on [int64], so the scripts are identical on every
   backend. Not for statistical quality. *)

type t = int64 ref

let make seed : t = ref (Int64.of_int seed)

(* In [0, bound). *)
let below (t : t) bound =
  t := Int64.add (Int64.mul !t 6364136223846793005L) 1442695040888963407L;
  let hi = Int64.to_int (Int64.shift_right_logical !t 33) in
  hi mod bound

(* A random valid script over int keys: [n] allocs, frees interleaved, some
   blocks never freed. Alignments are powers of two up to
   [2^max_log_alignment] (default 0: every block unaligned), drawn only when
   asked for, so a script without them is the one it always was. *)
let script ?(max_log_alignment = 0) (t : t) ~n ~max_size =
  let live = ref [] and events = ref [] and next = ref 0 in
  while !next < n || !live <> [] do
    let alloc = !next < n && (!live = [] || below t 3 <> 0) in
    if alloc then begin
      let key = !next in
      incr next;
      let size = Units.size (Int64.of_int (below t (max_size + 1))) in
      let alignment =
        if max_log_alignment = 0 then Units.one
        else
          Units.alignment
            (Int64.shift_left 1L (below t (max_log_alignment + 1)))
      in
      events := Interval_alloc.Event.Alloc { key; size; alignment } :: !events;
      live := key :: !live
    end
    else begin
      let i = below t (List.length !live) in
      let key = List.nth !live i in
      live := List.filter (fun k -> k <> key) !live;
      if below t 8 <> 0 then events := Interval_alloc.Event.Free key :: !events
    end
  done;
  List.rev !events
