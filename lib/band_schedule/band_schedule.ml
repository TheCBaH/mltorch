module Row = struct
  type t = int

  let of_int n = n
  let to_int n = n
  let compare = Int.compare
  let equal = Int.equal
  let pp = Fmt.int
end

module Count = struct
  type t = int

  let of_int n =
    if n < 0 then invalid_arg "Band_schedule.Count.of_int: negative" else n

  let to_int n = n
  let compare = Int.compare
  let pp = Fmt.int
end

module Span = struct
  type 'role t = int
  type block
  type chunk

  let positive what n =
    if n < 1 then invalid_arg ("Band_schedule.Span." ^ what ^ ": below 1")
    else n

  let block n = positive "block" n
  let chunk n = positive "chunk" n
  let to_int n = n
  let compare = Int.compare
  let pp = Fmt.int
end

module Axis = struct
  type t = {
    extent : Count.t;
    stride : Count.t;
    pad : Count.t;
    dilation : Count.t;
  }

  let window a = Count.of_int (((a.extent - 1) * a.dilation) + 1)
end

type kernel = {
  name : string;
  rows : Axis.t;
  cols : Axis.t;
  chunk : Span.chunk Span.t;
}

type event = {
  kernel : string;
  produced : Row.t * Row.t;
  consumed : Row.t * Row.t;
}

type t = { events : event list; windows : Count.t list }

let output_height k ~input =
  let a = k.rows in
  let w = Count.to_int (Axis.window a) in
  if input + (2 * a.Axis.pad) < w then 0
  else ((input + (2 * a.Axis.pad) - w) / a.Axis.stride) + 1

let input_rows k ~lo ~hi ~height =
  let r = k.rows in
  let a = (lo * r.Axis.stride) - r.Axis.pad in
  let b =
    ((hi - 1) * r.Axis.stride) - r.Axis.pad + Count.to_int (Axis.window r)
  in
  (max 0 a, min height b)

let invalid what = invalid_arg ("Band_schedule.run: " ^ what)

let run kernels ~input ~block =
  let ks = Array.of_list kernels in
  let n = Array.length ks in
  if n = 0 then invalid "an empty chain";
  if block = 0 then invalid "a zero block";
  let in_h = Array.make n 0 and out_h = Array.make n 0 in
  Array.iteri
    (fun i k ->
      in_h.(i) <- (if i = 0 then input else out_h.(i - 1));
      out_h.(i) <- output_height k ~input:in_h.(i);
      if out_h.(i) = 0 then invalid ("kernel " ^ k.name ^ " produces no rows"))
    ks;
  let chunk i = if i = n - 1 then block else ks.(i).chunk in
  let produced = Array.make n 0 in
  let events = ref [] and time = ref 0 in
  let rec ensure i upto =
    while produced.(i) < upto do
      let a = produced.(i) in
      let b = min (a + chunk i) out_h.(i) in
      let lo, hi = input_rows ks.(i) ~lo:a ~hi:b ~height:in_h.(i) in
      if i > 0 then ensure (i - 1) hi;
      events :=
        { kernel = ks.(i).name; produced = (a, b); consumed = (lo, hi) }
        :: !events;
      incr time;
      produced.(i) <- b
    done
  in
  ensure (n - 1) out_h.(n - 1);
  let events = List.rev !events in
  let ev = Array.of_list events in
  (* A producer's row is live from the event that makes it to the last event
     of its consumer that reads it. *)
  let windows =
    List.init n (fun j ->
        if j = n - 1 then 0
        else
          let rows = out_h.(j) in
          let born = Array.make rows (-1) and last = Array.make rows (-1) in
          Array.iteri
            (fun t e ->
              if String.equal e.kernel ks.(j).name then
                for r = fst e.produced to snd e.produced - 1 do
                  born.(r) <- t;
                  if last.(r) < t then last.(r) <- t
                done
              else if String.equal e.kernel ks.(j + 1).name then
                for r = fst e.consumed to snd e.consumed - 1 do
                  if last.(r) < t then last.(r) <- t
                done)
            ev;
          let peak = ref 0 in
          for t = 0 to Array.length ev - 1 do
            let live = ref 0 in
            for r = 0 to rows - 1 do
              if born.(r) <= t && t <= last.(r) then incr live
            done;
            if !live > !peak then peak := !live
          done;
          let c = ks.(j).chunk in
          (!peak + c - 1) / c * c)
  in
  { events; windows }

let pp ppf t =
  List.iter
    (fun e ->
      Fmt.pf ppf "%s produces [%d,%d) from [%d,%d)@." e.kernel (fst e.produced)
        (snd e.produced) (fst e.consumed) (snd e.consumed))
    t.events;
  Fmt.pf ppf "windows: %a@." Fmt.(list ~sep:(any " ") int) t.windows

type refusal =
  | Fan_out of string
  | Rows_not_demanded of { kernel : string; row : Row.t }
  | Second_sink of string
  | Unknown_kernel of string

let pp_refusal ppf = function
  | Fan_out k -> Fmt.pf ppf "%s feeds two consumers in the segment" k
  | Rows_not_demanded { kernel; row } ->
      Fmt.pf ppf "a side output of %s: row %d is never demanded" kernel row
  | Second_sink k -> Fmt.pf ppf "%s is a second sink" k
  | Unknown_kernel k -> Fmt.pf ppf "%s is no kernel of the segment" k

let admit kernels ~consumers ~side_outputs ~input ~block =
  if kernels = [] then invalid "an empty chain";
  let names = List.map (fun k -> k.name) kernels in
  let known k = List.mem k names in
  let last = List.nth names (List.length names - 1) in
  let ( let* ) = Result.bind in
  let* () =
    List.fold_left
      (fun acc k ->
        let* () = acc in
        let cs = Option.value ~default:[] (List.assoc_opt k consumers) in
        match cs with
        | [] -> if String.equal k last then Ok () else Error (Second_sink k)
        | [ _ ] -> Ok ()
        | _ :: _ :: _ -> Error (Fan_out k))
      (Ok ()) names
  in
  let* () =
    match List.find_opt (fun k -> not (known k)) side_outputs with
    | Some k -> Error (Unknown_kernel k)
    | None -> Ok ()
  in
  let s = run kernels ~input ~block in
  let ks = Array.of_list kernels in
  let heights = Array.make (Array.length ks) 0 in
  Array.iteri
    (fun i k ->
      let in_h = if i = 0 then input else heights.(i - 1) in
      heights.(i) <- output_height k ~input:in_h)
    ks;
  let index k =
    let rec go i = if String.equal ks.(i).name k then i else go (i + 1) in
    go 0
  in
  let* () =
    List.fold_left
      (fun acc k ->
        let* () = acc in
        let j = index k in
        if j = Array.length ks - 1 then Ok ()
        else
          let demanded = Array.make heights.(j) false in
          List.iter
            (fun e ->
              if String.equal e.kernel ks.(j + 1).name then
                for r = fst e.consumed to snd e.consumed - 1 do
                  demanded.(r) <- true
                done)
            s.events;
          let rec first r =
            if r >= Array.length demanded then Ok ()
            else if demanded.(r) then first (r + 1)
            else Error (Rows_not_demanded { kernel = k; row = r })
          in
          first 0)
      (Ok ()) side_outputs
  in
  Ok s

type violation =
  | Consumed_not_produced of { event : Count.t; kernel : string; row : Row.t }
  | Consumed_range_wrong of { event : Count.t; kernel : string }
  | Row_gap of { kernel : string; row : Row.t }
  | Row_recomputed of { kernel : string; row : Row.t }
  | Rows_missing of { kernel : string; row : Row.t }
  | Unknown_event_kernel of string

let pp_violation ppf = function
  | Consumed_not_produced { event; kernel; row } ->
      Fmt.pf ppf "event %d of %s reads row %d before it is produced" event
        kernel row
  | Consumed_range_wrong { event; kernel } ->
      Fmt.pf ppf "event %d of %s reads the wrong input rows" event kernel
  | Row_gap { kernel; row } -> Fmt.pf ppf "%s skips to row %d" kernel row
  | Row_recomputed { kernel; row } ->
      Fmt.pf ppf "%s produces row %d twice" kernel row
  | Rows_missing { kernel; row } ->
      Fmt.pf ppf "%s never produces row %d" kernel row
  | Unknown_event_kernel k -> Fmt.pf ppf "an event names %s, no kernel" k

let verify kernels ~input t =
  let ks = Array.of_list kernels in
  let n = Array.length ks in
  let index name =
    let rec go i =
      if i >= n then None
      else if String.equal ks.(i).name name then Some i
      else go (i + 1)
    in
    go 0
  in
  let in_h = Array.make n 0 and out_h = Array.make n 0 in
  Array.iteri
    (fun i k ->
      in_h.(i) <- (if i = 0 then input else out_h.(i - 1));
      out_h.(i) <- output_height k ~input:in_h.(i))
    ks;
  let produced = Array.make n 0 in
  let ( let* ) = Result.bind in
  let rec events t = function
    | [] -> Ok ()
    | e :: rest -> (
        match index e.kernel with
        | None -> Error (Unknown_event_kernel e.kernel)
        | Some i ->
            let a, b = e.produced in
            let* () =
              if a < produced.(i) then
                Error (Row_recomputed { kernel = e.kernel; row = a })
              else if a > produced.(i) then
                Error (Row_gap { kernel = e.kernel; row = a })
              else Ok ()
            in
            let want = input_rows ks.(i) ~lo:a ~hi:b ~height:in_h.(i) in
            let* () =
              if e.consumed <> want then
                Error (Consumed_range_wrong { event = t; kernel = e.kernel })
              else Ok ()
            in
            let* () =
              if i > 0 && snd e.consumed > produced.(i - 1) then
                Error
                  (Consumed_not_produced
                     { event = t; kernel = e.kernel; row = produced.(i - 1) })
              else Ok ()
            in
            produced.(i) <- b;
            events (t + 1) rest)
  in
  let* () = events 0 t.events in
  let rec missing i =
    if i >= n then Ok ()
    else if produced.(i) < out_h.(i) then
      Error (Rows_missing { kernel = ks.(i).name; row = produced.(i) })
    else missing (i + 1)
  in
  missing 0

type mutation =
  | Drop_event
  | Repeat_event
  | Run_consumer_first
  | Shift_consumed

let mutate m t =
  let events = t.events in
  let last_kernel = (List.nth events (List.length events - 1)).kernel in
  let first_of k = List.find (fun e -> String.equal e.kernel k) events in
  let remove_first k l =
    let rec go seen = function
      | [] -> []
      | e :: rest ->
          if (not seen) && String.equal e.kernel k then rest
          else e :: go seen rest
    in
    go false l
  in
  let events =
    match m with
    | Drop_event -> remove_first last_kernel events
    | Repeat_event ->
        let f = first_of (List.hd events).kernel in
        f :: events
    | Run_consumer_first ->
        first_of last_kernel :: remove_first last_kernel events
    | Shift_consumed ->
        let f = first_of last_kernel in
        let a, b = f.consumed in
        List.map
          (fun e -> if e == f then { e with consumed = (a + 1, b) } else e)
          events
  in
  { t with events }

module Ring_row = struct
  type t = int

  let to_int n = n
  let compare = Int.compare
  let pp = Fmt.int
end

module Window = struct
  type t = { rows : Count.t; chunk : Span.chunk Span.t }

  let make ~rows ~chunk =
    if rows = 0 || chunk = 0 || rows mod chunk <> 0 then
      invalid_arg "Band_schedule.Window.make: rows is not a multiple of chunk";
    { rows; chunk }

  let rows t = t.rows
  let chunk t = t.chunk

  let ring_row t r =
    if r < 0 then invalid_arg "Band_schedule.Window.ring_row: negative row";
    r mod t.rows

  let spans_wrap t ~lo ~hi = hi > lo && ring_row t lo + (hi - lo) > t.rows
end
