(* Section 1: the walk_core foundation -- the seeded RNG and the bit-exact f32
   spelling. Everything here is reachable from every backend, including melange
   (walk_core depends on jsont alone, never Jsont_bytesrw).

   The PCG draws are the reason this section exists: [Pcg.next] spans
   [0, 2^32), which does not fit a non-negative OCaml [int] under js_of_ocaml,
   and this prints enough draws to cross 2^31. The f32 values are the awkward
   ones -- a decimal that is not exactly representable, a subnormal, and the
   three patterns that fall back to a hex spelling. *)

let awkward =
  [
    1e-5;
    1e-8;
    0.1;
    Float.pi;
    1.0 /. 3.0;
    1e30;
    1.4e-45 (* subnormal *);
    -0.0;
    Float.infinity;
    Float.nan;
  ]

let run () =
  print_endline "=== walk_core ===";
  let pcg = ref (Walk_core.Pcg.seed ~seed:42L ~seq:1L) in
  for _ = 1 to 8 do
    let n, next = Walk_core.Pcg.next !pcg in
    pcg := next;
    Printf.printf "pcg %Lu\n" n
  done;
  let u, after = Walk_core.Pcg.uniform ~low:(-1.0) ~high:1.0 !pcg in
  let g, _ = Walk_core.Pcg.normal ~mean:0.0 ~std:1.0 after in
  Printf.printf "uniform %s normal %s\n"
    (Walk_core.Float32.to_hex u)
    (Walk_core.Float32.to_hex g);
  List.iter
    (fun x ->
      let f = Walk_core.Float32.to_f32 x in
      (* [enc_json] is specified to yield a number or a hex string and nothing
         else. Anything else is a defect, and rendering it as text would
         reproduce on both backends, diff clean and pass -- so raise. *)
      let spelled =
        match Walk_core.Float32.enc_json f with
        | Jsont.Number (n, _) -> Printf.sprintf "number %.17g" n
        | Jsont.String (s, _) -> Printf.sprintf "hex %s" s
        | _ ->
            failwith
              (Printf.sprintf
                 "probe: enc_json %s produced neither a number nor a hex string"
                 (Walk_core.Float32.to_hex f))
      in
      Printf.printf "f32 %s -> %s\n" (Walk_core.Float32.to_hex f) spelled)
    awkward
