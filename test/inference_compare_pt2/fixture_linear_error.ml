(* Where does a one-linear fixture's output leave the producer's? The SmolVLM
   connector is a pixel shuffle (pure data movement) and one bias-free linear,
   so every output element is one dot product of 12288 terms. This recomputes
   each in compensated binary64 from the case input and the checkpoint weight
   and measures the producer's reference, the ATen output and both Native
   accumulation policies (dumped by fixture_pair_run) against it.
   Usage: fixture_linear_error COHORT CACHE ARTIFACT CASE DUMP_DIR *)
module P = Inference_compare_pt2
module U = Pt2_fixture_unix

let hashers = Inference_compare_sha256.registry
let algorithm = Inference_compare_sha256.algorithm
let ok = function Ok x -> x | Error m -> failwith m

let unwrap = function
  | Ok x -> x
  | Error e -> failwith (Fmt.str "%a" U.Fixture.pp_error e)

let f32s_of_bytes s =
  Array.init
    (String.length s / 4)
    (fun i -> Int32.float_of_bits (String.get_int32_le s (4 * i)))

let f32s_of_tensor (t : Pt2_tensor.t) =
  match Pt2_fixture.Logical.of_pt2 t with
  | Error _ -> failwith "logical tensor"
  | Ok l ->
      f32s_of_bytes
        (String.init (Pt2_storage.length l.data) (fun i ->
             Char.chr (Pt2_storage.get_uint8 l.data i)))

let read path = In_channel.with_open_bin path In_channel.input_all

(* Neumaier-compensated sum of the (exact in binary64) products. *)
let dot_exact x w =
  let s = ref 0.0 and c = ref 0.0 and cond = ref 0.0 in
  for j = 0 to Array.length x - 1 do
    let p = x.(j) *. w.(j) in
    cond := !cond +. Float.abs p;
    let t = !s +. p in
    if Float.abs !s >= Float.abs p then c := !c +. (!s -. t +. p)
    else c := !c +. (p -. t +. !s);
    s := t
  done;
  (!s +. !c, !cond)

let () =
  let cohort_path = Sys.argv.(1)
  and cache_dir = Sys.argv.(2)
  and artifact = Sys.argv.(3)
  and case = Sys.argv.(4)
  and dump = Sys.argv.(5) in
  let cohort =
    unwrap
      (Err.payload
         (let open Err.Syntax in
          let* text = U.Fetch.read cohort_path in
          Pt2_fixture.Cohort.of_string text))
  in
  let entry = Option.get (Pt2_fixture.Cohort.find cohort artifact) in
  let cache = unwrap (Err.payload (U.Cache.create cache_dir)) in
  let fx =
    unwrap (Err.payload (U.Fixture.open_ (U.Bundle.config cache) cohort entry))
  in
  let tensors role =
    match
      Pt2_archive.load_pt_tensor_map
        (Filename.concat fx.bundle.dir
           (Printf.sprintf "cases/%s/%s.pt" case role))
    with
    | Ok [ (_, t) ] -> f32s_of_tensor t
    | _ -> failwith role
  in
  let hidden = tensors "inputs" and reference = tensors "outputs" in
  let identity_text =
    unwrap (Err.payload (U.Bundle.read_member fx.bundle "models/model.json"))
  in
  let aten =
    ok (P.Aten_model.of_archive ~hashers ~algorithm ~identity_text fx.archive)
  in
  let target =
    match
      List.find_map (fun (s : P.Aten_model.source) -> s.captured) aten.sources
    with
    | Some t -> t
    | None -> failwith "no weight"
  in
  let weight =
    f32s_of_tensor
      (match Pt2_archive.load_captured_tensor fx.archive target with
      | Ok t -> t
      | Error _ -> failwith "weight")
  in
  let rows = 64 and k = 12288 and outs = 576 in
  assert (Array.length weight = outs * k);
  assert (Array.length hidden = 1024 * 768);
  (* view [1,32,32,768] -> [1,32,8,3072] -> permute -> [1,8,8,12288] ->
     permute -> rows (ho, wo): column j = ki*3072 + kw*768 + c reads the token
     (ho*4+ki, wo*4+kw). *)
  let x = Array.make (rows * k) 0.0 in
  for ho = 0 to 7 do
    for wo = 0 to 7 do
      for ki = 0 to 3 do
        for kw = 0 to 3 do
          for c = 0 to 767 do
            let token = (((ho * 4) + ki) * 32) + (wo * 4) + kw in
            x.((((ho * 8) + wo) * k) + (ki * 3072) + (kw * 768) + c) <-
              hidden.((token * 768) + c)
          done
        done
      done
    done
  done;
  let exact = Array.make (rows * outs) 0.0
  and cond = Array.make (rows * outs) 0.0 in
  for r = 0 to rows - 1 do
    let xr = Array.sub x (r * k) k in
    for o = 0 to outs - 1 do
      let v, c = dot_exact xr (Array.sub weight (o * k) k) in
      exact.((r * outs) + o) <- v;
      cond.((r * outs) + o) <- c
    done
  done;
  let sources =
    [
      ("producer (PyTorch)", reference);
      ( "ATen C++ (local)",
        f32s_of_bytes
          (read (Filename.concat dump ("aten-" ^ case ^ "/final-0.bin"))) );
      ( "Native binary64 dots",
        f32s_of_bytes
          (read (Filename.concat dump ("nat-exact-" ^ case ^ "/final-0.bin")))
      );
      ( "Native f32 sequential",
        f32s_of_bytes
          (read (Filename.concat dump ("nat-seq-" ^ case ^ "/final-0.bin"))) );
    ]
  in
  let n = rows * outs in
  let round32 v = Int32.float_of_bits (Int32.bits_of_float v) in
  Printf.printf "%s %s: %d outputs, each a %d-term dot\n" artifact case n k;
  Printf.printf "%-24s %12s %12s %12s %10s %10s\n" "source" "max|e|" "rms e"
    "bias" "= rounded" "viol/ref";
  List.iter
    (fun (name, y) ->
      let mx = ref 0.0
      and ss = ref 0.0
      and bias = ref 0.0
      and same = ref 0
      and viol = ref 0 in
      for i = 0 to n - 1 do
        let e = y.(i) -. exact.(i) in
        mx := Float.max !mx (Float.abs e);
        ss := !ss +. (e *. e);
        bias := !bias +. e;
        if y.(i) = round32 exact.(i) then incr same;
        if
          Float.abs (y.(i) -. reference.(i))
          > 1e-5 +. (1e-4 *. Float.abs reference.(i))
        then incr viol
      done;
      Printf.printf "%-24s %12.4g %12.4g %12.4g %9.2f%% %10d\n" name !mx
        (sqrt (!ss /. float n))
        (!bias /. float n)
        (100.0 *. float !same /. float n)
        !viol)
    sources;
  (let same = ref 0 in
   let seq = snd (List.nth sources 3) in
   Array.iteri
     (fun i v ->
       if Int32.equal (Int32.bits_of_float v) (Int32.bits_of_float seq.(i)) then
         incr same)
     reference;
   Printf.printf
     "\nproducer output bitwise equal to Native f32 sequential: %d of %d\n"
     !same n);
  (* The elements any source takes outside the published tolerance of the
     producer's output. *)
  Printf.printf "\nelements outside atol 1e-5 + rtol 1e-4*|producer|:\n";
  Printf.printf "%-8s %12s %12s %10s  %s\n" "(row,o)" "producer" "exact"
    "sum|xw|" "error vs exact: producer / ATen / Native64 / Native32seq";
  for i = 0 to n - 1 do
    if
      List.exists
        (fun (_, y) ->
          Float.abs (y.(i) -. reference.(i))
          > 1e-5 +. (1e-4 *. Float.abs reference.(i)))
        sources
    then
      Printf.printf "(%2d,%3d) %12.6g %12.6g %10.4g  %s\n" (i / outs)
        (i mod outs) reference.(i) exact.(i) cond.(i)
        (String.concat " / "
           (List.map
              (fun (_, y) -> Printf.sprintf "%.3g" (y.(i) -. exact.(i)))
              sources))
  done
