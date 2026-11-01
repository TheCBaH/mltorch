(* Lawson-Hanson NNLS over a small dense system. *)

let solve_ls a b cols =
  (* least squares on the columns [cols] of [a] by the normal equations with a
     tiny ridge, Gaussian elimination with partial pivoting *)
  let k = List.length cols in
  let cols = Array.of_list cols in
  let n = Array.length a in
  let m = Array.make_matrix k (k + 1) 0. in
  for i = 0 to k - 1 do
    for j = 0 to k - 1 do
      let s = ref 0. in
      for r = 0 to n - 1 do
        s := !s +. (a.(r).(cols.(i)) *. a.(r).(cols.(j)))
      done;
      m.(i).(j) <- !s
    done;
    m.(i).(i) <- (m.(i).(i) *. (1. +. 1e-12)) +. 1e-18;
    let s = ref 0. in
    for r = 0 to n - 1 do
      s := !s +. (a.(r).(cols.(i)) *. b.(r))
    done;
    m.(i).(k) <- !s
  done;
  for p = 0 to k - 1 do
    let best = ref p in
    for r = p + 1 to k - 1 do
      if Float.abs m.(r).(p) > Float.abs m.(!best).(p) then best := r
    done;
    let t = m.(p) in
    m.(p) <- m.(!best);
    m.(!best) <- t;
    let piv = m.(p).(p) in
    if piv <> 0. then
      for r = 0 to k - 1 do
        if r <> p then
          let fct = m.(r).(p) /. piv in
          for c = p to k do
            m.(r).(c) <- m.(r).(c) -. (fct *. m.(p).(c))
          done
      done
  done;
  Array.init k (fun i -> if m.(i).(i) = 0. then 0. else m.(i).(k) /. m.(i).(i))

let fit ?(priors = []) ?(weight = 1.) samples =
  let present n = List.exists (fun (fs, _) -> List.mem_assoc n fs) samples in
  (* each row a relative miss times its weight: one for a sample, [weight] for
     a prior, which pulls its rate toward the cycles observed *)
  let rows =
    List.map (fun (fs, y) -> (fs, y, 1.)) samples
    @ List.filter_map
        (fun (n, c) ->
          if present n && c > 0. then Some ([ (n, 1.) ], c, weight) else None)
        priors
  in
  let names =
    List.sort_uniq compare
      (List.concat_map (fun (fs, _, _) -> List.map fst fs) rows)
  in
  let names = Array.of_list names in
  let nf = Array.length names in
  let idx = Hashtbl.create nf in
  Array.iteri (fun i n -> Hashtbl.replace idx n i) names;
  let a =
    Array.of_list
      (List.map
         (fun (fs, y, w) ->
           let row = Array.make nf 0. in
           List.iter (fun (k, v) -> row.(Hashtbl.find idx k) <- w *. v /. y) fs;
           row)
         rows)
  in
  let b = Array.of_list (List.map (fun (_, _, w) -> w) rows) in
  let n = Array.length a in
  let x = Array.make nf 0. in
  let passive = Array.make nf false in
  let gradient () =
    Array.init nf (fun j ->
        let s = ref 0. in
        for r = 0 to n - 1 do
          let pred = ref 0. in
          for c = 0 to nf - 1 do
            pred := !pred +. (a.(r).(c) *. x.(c))
          done;
          s := !s +. (a.(r).(j) *. (b.(r) -. !pred))
        done;
        !s)
  in
  let rec outer iter =
    if iter < 3 * nf then (
      let w = gradient () in
      let best = ref (-1) in
      Array.iteri
        (fun j wj ->
          if (not passive.(j)) && wj > 1e-12 && (!best < 0 || wj > w.(!best))
          then best := j)
        w;
      if !best >= 0 then (
        passive.(!best) <- true;
        let rec inner guard =
          let cols = List.filter (fun j -> passive.(j)) (List.init nf Fun.id) in
          let z = solve_ls a b cols in
          let zc = List.combine cols (Array.to_list z) in
          if List.for_all (fun (_, v) -> v > 0.) zc || guard > 3 * nf then
            List.iter (fun (j, v) -> x.(j) <- Float.max 0. v) zc
          else
            let alpha =
              List.fold_left
                (fun acc (j, v) ->
                  if v <= 0. then Float.min acc (x.(j) /. (x.(j) -. v)) else acc)
                1. zc
            in
            List.iter
              (fun (j, v) -> x.(j) <- x.(j) +. (alpha *. (v -. x.(j))))
              zc;
            List.iter
              (fun (j, _) ->
                if x.(j) <= 1e-12 then (
                  passive.(j) <- false;
                  x.(j) <- 0.))
              zc;
            inner (guard + 1)
        in
        inner 0;
        outer (iter + 1)))
  in
  outer 0;
  Array.to_list (Array.mapi (fun i n -> (n, x.(i))) names)
  |> List.filter (fun (_, v) -> v > 0.)
