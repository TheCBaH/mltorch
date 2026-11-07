(* The fast form of [exp] (and so [erf]) against the reference, bit for bit,
   over every one of the 2^32 binary32 inputs, as native C: both are printed
   from {!F32_math} as C functions of [const float] statements (one rounding
   each, nothing fused), the fast pieces as [truncf] and the bits of
   [n + 1.5 * 2^23 + 127] shifted into the exponent field, compiled with the
   host compiler and run. A NaN is compared by its bits too.

   Usage: exhaustive_c.exe [--bias N] [--cc CC]. [--bias 128] instead of the
   correct 127 is the mutation the run must catch. Exits 1 on any difference. *)

let lit x =
  let x = F32_math.round32 x in
  if Float.is_integer x && Float.abs x < 1e9 then Printf.sprintf "%.1ff" x
  else Printf.sprintf "%hf" x

(* The C text of [exp] or [erf] under [Make] or [Make_fast]. *)
let fn ~name ~fast ~bias which =
  let buf = Buffer.create 4096 and n = ref 0 in
  let bind e =
    incr n;
    let v = Printf.sprintf "t%d" !n in
    Buffer.add_string buf (Printf.sprintf "  const float %s = %s;\n" v e);
    v
  in
  let module O = struct
    type t = string

    let const x = lit x
    let add a b = bind (Printf.sprintf "%s + %s" a b)
    let sub a b = bind (Printf.sprintf "%s - %s" a b)
    let mul a b = bind (Printf.sprintf "%s * %s" a b)
    let div a b = bind (Printf.sprintf "%s / %s" a b)
    let select_lt a b c d = bind (Printf.sprintf "%s < %s ? %s : %s" a b c d)
    let select_eq a b c d = bind (Printf.sprintf "%s == %s ? %s : %s" a b c d)
    let share x = x
  end in
  let module F = struct
    type t = string

    let pow2_int n =
      bind
        (Printf.sprintf "pow2_bits(%s + %s)" n
           (lit (12582912. +. float_of_int bias)))

    let trunc_small x = bind (Printf.sprintf "truncf(%s)" x)
  end in
  let f =
    if fast then
      let module M = F32_math.Make_fast (O) (F) in
      match which with `Erf -> M.erf | `Exp -> M.exp
    else
      let module M = F32_math.Make (O) in
      match which with `Erf -> M.erf | `Exp -> M.exp
  in
  let r = f "x" in
  Printf.sprintf "static float %s(float x) {\n%s  return %s;\n}\n" name
    (Buffer.contents buf) r

let source ~bias =
  String.concat "\n"
    [
      "#include <math.h>\n\
       #include <stdint.h>\n\
       #include <stdio.h>\n\
       #include <string.h>\n";
      "static inline float pow2_bits(float b) {\n\
      \  uint32_t u;\n\
      \  memcpy(&u, &b, sizeof u);\n\
      \  u <<= 23;\n\
      \  float f;\n\
      \  memcpy(&f, &u, sizeof f);\n\
      \  return f;\n\
       }\n";
      fn ~name:"exp_ref" ~fast:false ~bias `Exp;
      fn ~name:"exp_fast" ~fast:true ~bias `Exp;
      fn ~name:"erf_ref" ~fast:false ~bias `Erf;
      fn ~name:"erf_fast" ~fast:true ~bias `Erf;
      "static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, sizeof u); \
       return u; }\n\n\
       int main(void) {\n\
      \  unsigned long long differ = 0;\n\
      \  uint32_t first = 0;\n\
      \  uint32_t u = 0;\n\
      \  do {\n\
      \    float x;\n\
      \    memcpy(&x, &u, sizeof x);\n\
      \    if (bits(exp_ref(x)) != bits(exp_fast(x)) ||\n\
      \        bits(erf_ref(x)) != bits(erf_fast(x))) {\n\
      \      if (differ == 0) first = u;\n\
      \      differ++;\n\
      \    }\n\
      \    u++;\n\
      \  } while (u != 0);\n\
      \  printf(\"2^32 inputs, exp and erf; differing: %llu\", differ);\n\
      \  if (differ) printf(\" (first 0x%08x)\", first);\n\
      \  printf(\"\\n\");\n\
      \  return differ != 0;\n\
       }\n";
    ]

let () =
  let bias = ref 127 and cc = ref "cc" in
  Arg.parse
    [
      ("--bias", Arg.Set_int bias, "N the exponent bias (127; 128 is wrong)");
      ("--cc", Arg.Set_string cc, "CC the C compiler (cc)");
    ]
    (fun _ -> ())
    "exhaustive_c.exe [--bias N] [--cc CC]";
  let c = Filename.temp_file "f32_exhaustive" ".c" in
  let exe = Filename.remove_extension c in
  Out_channel.with_open_text c (fun oc -> output_string oc (source ~bias:!bias));
  let compile =
    Printf.sprintf
      "%s -O2 -std=c11 -ffp-contract=off -fno-fast-math -o %s %s -lm"
      (Filename.quote_command !cc [])
      (Filename.quote exe) (Filename.quote c)
  in
  if Sys.command compile <> 0 then (
    prerr_endline "exhaustive_c: the C compiler failed";
    exit 2);
  let t0 = Unix.gettimeofday () in
  let rc = Sys.command (Filename.quote exe) in
  Printf.printf "%.1f s\n" (Unix.gettimeofday () -. t0);
  exit (if rc = 0 then 0 else 1)
