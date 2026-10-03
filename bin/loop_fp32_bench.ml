(* Benchmark of the working-precision configurations on dense kernels: what
   binary32 SIMD buys over binary64, kernel by kernel, on native C (default) or on
   direct Wasm under node ([--wasm]).

   Each program is hand-lowered Loop IR (matvec and matmul with weights in [K,N]
   order and independent output channels contiguous, and a pointwise kernel),
   compiled four ways with the host compiler:

     f64_scalar         Reference_f64, no vectors
     f64_simd           Reference_f64, strict binary64 vectors (inner loops on)
     fp32_scalar        forced binary32, no vectors
     fp32_ordered_simd  Simd_fp32_ordered, binary32 vectors, each output's order

   Every configuration is timed inside the generated binary (nine batches of
   REPS calls after a warm-up, ms per batch, all samples kept) and verified
   independently of its timing: the output must be bitwise the interpreter's at
   the configuration's own precision, so a wrong result fails the run whatever
   it measured. [--selftest] corrupts one cell of one result and requires the
   verification to reject it.

   [--wasm] runs the same programs through the direct Wasm backend under node
   instead: the module is lowered for [Loop_target.wasm128] (and, when node
   validates the relaxed-simd probe, [wasm128_relaxed], whose multiply-add the
   engine may fuse, so its result is checked as one of the two admissible
   answers), instantiated once, warmed for at least 100 ms so the engine has
   tiered up, then timed in the same nine batches inside node. The module's own
   memory holds the buffers, so the timed region is the kernel alone.

   Output is one JSON object per line on stdout (also written to FILE with
   [--json=FILE], which CI keeps as an artifact); a table goes to stderr. The exit
   status reflects verification only, never timing, so a noisy runner cannot fail it.
   Inputs come from a fixed LCG (seed 17) scaled into about [-1, 1]. Setup
   (code generation, compilation, packing) is outside the timed region. *)

open Loop_ir

module Bench = struct
  type t = { name : string; program : Loop_program.t; reps : int }
end

(* ---- the programs -------------------------------------------------------- *)

let shape_w n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:n ~c:1
let f32 = Payload.Fmt Payload.F32
let tid = Tensor_id.of_int

let buffer id n role =
  let sg =
    Tensor_sig.create ~id:(tid id) ~name:"" ~shape:(shape_w n) ~fmt:f32 ()
  in
  { Loop_buffer.id = tid id; sg; role }

let v n = Loop_var.of_int n
let var n = Loop_index.Var (v n)
let const n = Loop_index.Const n
let scale k i = Loop_index.Scale (k, i)
let ( +: ) a b = Loop_index.Add (a, b)
let acc = Loop_temp.of_int 0
let acc_e = Loop_expr.Temp (Loop_carrier.Float, acc)
let ld b i = Loop_expr.Load_flat (b, i)

let for_ i hi body =
  Loop_stmt.For { var = v i; lo = const 0; hi = const hi; body }

let program buffers body =
  {
    Loop_program.buffers;
    body;
    scan_limits = Expr.Scan_limits.default;
    max_depth = 64;
  }

(* out[m, n] = sum_k a[m, k] * w[k, n], [m] rows (1 for a matvec). [~weights] is
   where the weights sit: [`Kn] (K rows of N, contiguous across the outputs, what
   an output-axis vector loads cheaply) or [`Nk] (N rows of K, contiguous along
   the sum, a PyTorch convolution's own layout: an output-axis vector gathers). *)
let matmul ?(weights = `Kn) ~name ~m ~k ~n ~reps () =
  let a = buffer 0 (m * k) Loop_buffer.Input
  and w = buffer 1 (k * n) Loop_buffer.Input
  and out = buffer 2 (m * n) Loop_buffer.Output in
  let mi = var 0 and ni = var 1 and ki = var 2 in
  let term =
    Loop_expr.Binary
      ( Expr.Value.Mul,
        ld a (scale k mi +: ki),
        ld w
          (match weights with
          | `Kn -> scale n ki +: ni
          | `Nk -> scale k ni +: ki) )
  in
  let cell =
    [
      Loop_stmt.Assign (Loop_carrier.Float, acc, Loop_expr.Const 0.);
      for_ 2 k
        [
          Loop_stmt.Mark Loop_mark.Reduction;
          Loop_stmt.Assign
            ( Loop_carrier.Float,
              acc,
              Loop_expr.Binary (Expr.Value.Add, acc_e, term) );
        ];
      Loop_stmt.Store_flat
        {
          buffer = out;
          offset = scale n mi +: ni;
          value = Loop_stored.F32 (Loop_expr.Round_f32 acc_e);
        };
    ]
  in
  {
    Bench.name;
    reps;
    program = program [ a; w; out ] [ for_ 0 m [ for_ 1 n cell ] ];
  }

let pointwise ~name ~n ~reps =
  let x = buffer 0 n Loop_buffer.Input
  and out = buffer 1 n Loop_buffer.Output in
  let i = var 0 in
  let e =
    Loop_expr.Float_max
      ( Loop_expr.Binary
          ( Expr.Value.Add,
            Loop_expr.Binary (Expr.Value.Mul, ld x i, Loop_expr.Const 1.5),
            Loop_expr.Const (-0.25) ),
        Loop_expr.Const 0. )
  in
  {
    Bench.name;
    reps;
    program =
      program [ x; out ]
        [
          for_ 0 n
            [
              Loop_stmt.Store_flat
                {
                  buffer = out;
                  offset = i;
                  value = Loop_stored.F32 (Loop_expr.Round_f32 e);
                };
            ];
        ];
  }

let benches =
  [
    matmul ~name:"matvec k512 n128" ~m:1 ~k:512 ~n:128 ~reps:512 ();
    matmul ~name:"matvec k512 n16" ~m:1 ~k:512 ~n:16 ~reps:2048 ();
    matmul ~name:"matmul m16 k64 n128" ~m:16 ~k:64 ~n:128 ~reps:256 ();
    (* a 1x1 convolution's shape: many spatial rows, a short K, a wide N *)
    matmul ~name:"pointwise conv m196 k96 n96" ~m:196 ~k:96 ~n:96 ~reps:8 ();
    matmul ~weights:`Nk ~name:"matvec k512 n128, weights [N,K]" ~m:1 ~k:512
      ~n:128 ~reps:512 ();
    matmul ~weights:`Nk ~name:"pointwise conv m196 k96 n96, weights [N,K]"
      ~m:196 ~k:96 ~n:96 ~reps:8 ();
    matmul ~name:"matvec k1024 n4" ~m:1 ~k:1024 ~n:4 ~reps:4096 ();
    matmul ~name:"dot k4096" ~m:1 ~k:4096 ~n:1 ~reps:2048 ();
    pointwise ~name:"pointwise n4096" ~n:4096 ~reps:4096;
  ]

(* ---- configurations ------------------------------------------------------ *)

module Config = struct
  type t = {
    name : string;
    numerics : Loop_numerics.t;
    precision : Loop_numerics.Precision.t option;  (** forced, else resolved *)
    vector : Loop_target.t option;
  }

  let simd = Loop_target.with_inner_loops true Loop_target.neon128

  let native =
    [
      {
        name = "f64_scalar";
        numerics = Loop_numerics.Reference_f64;
        precision = None;
        vector = None;
      };
      {
        name = "f64_simd";
        numerics = Loop_numerics.Reference_f64;
        precision = None;
        vector = Some simd;
      };
      {
        name = "fp32_scalar";
        numerics = Loop_numerics.Simd_fp32_ordered;
        precision = Some Loop_numerics.Precision.F32;
        vector = None;
      };
      {
        name = "fp32_ordered_simd";
        numerics = Loop_numerics.Simd_fp32_ordered;
        precision = None;
        vector = Some simd;
      };
      {
        name = "fp32_relaxed_simd";
        numerics = Loop_numerics.Simd_fp32_relaxed;
        precision = None;
        vector = Some simd;
      };
    ]

  (* The same five on [wasm128] (inner loops are on there already), and one for
     relaxed SIMD where node validates its probe. *)
  let wasm () =
    let w = Loop_target.wasm128 in
    List.map
      (fun c -> if c.vector = None then c else { c with vector = Some w })
      native
    @
    if Loop_wasm_exec.Node.supports Wasm_features.Relaxed_simd then
      [
        {
          name = "fp32_relaxed_madd_simd";
          numerics = Loop_numerics.Simd_fp32_relaxed;
          precision = None;
          vector = Some Loop_target.wasm128_relaxed;
        };
      ]
    else []

  let all backend = match backend with `C -> native | `Wasm -> wasm ()
end

let backend =
  if Array.exists (String.equal "--wasm") Sys.argv then `Wasm else `C

(* ---- inputs -------------------------------------------------------------- *)

(* A fixed LCG (seed 17), scaled into about [-1, 1], one stream per buffer. *)
let lcg_stream ~seed n =
  let s = ref (Int64.of_int ((seed * 7919) + 17)) in
  Array.init n (fun _ ->
      (s :=
         Int64.(
           logand
             (add (mul !s 6364136223846793005L) 1442695040888963407L)
             Int64.max_int));
      let x = Int64.to_int (Int64.shift_right_logical !s 40) land 0xFFFFFF in
      (float_of_int x /. float_of_int 0x7FFFFF) -. 1.)

let bind (p : Loop_program.t) id =
  List.find_map
    (fun (b : Loop_buffer.t) ->
      if
        b.Loop_buffer.role = Loop_buffer.Input
        && Tensor_id.equal b.Loop_buffer.id id
      then
        let n =
          Dim.to_int (Vec6.get b.Loop_buffer.sg.Tensor_sig.shape Expr.Axis.W)
        in
        let data = lcg_stream ~seed:(Tensor_id.to_int id) n in
        Some
          (Tensor.materialize b.Loop_buffer.sg.Tensor_sig.shape (fun c ->
               Int32.float_of_bits
                 (Int32.bits_of_float data.((Vec6.get c Expr.Axis.W :> int)))))
      else None)
    p.Loop_program.buffers

(* ---- the timed binary ---------------------------------------------------- *)

let samples = 9

let timed_source (b : Bench.t) (k : Loop_c.t) =
  let p = b.Bench.program in
  let offsets, total = Loop_c_exec.layout p in
  let args =
    List.map2
      (fun ty off -> Printf.sprintf "(%s *)(blob + %d)" ty off)
      k.Loop_c.buffer_types offsets
  in
  let call =
    Printf.sprintf "kernel(&err, local%s)"
      (String.concat "" (List.map (fun a -> ", " ^ a) args))
  in
  let main =
    String.concat "\n"
      [
        "#include <stdio.h>";
        "#include <stdlib.h>";
        "#include <time.h>";
        "static double now_ms(void) {";
        "  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);";
        "  return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;";
        "}";
        "int main(int argc, char **argv) {";
        "  if (argc != 3) return 2;";
        Printf.sprintf "  const size_t total = %d;" total;
        "  unsigned char *blob = malloc(total ? total : 1);";
        "  FILE *in = fopen(argv[1], \"rb\");";
        "  if (!blob || !in || fread(blob, 1, total, in) != total) return 3;";
        "  fclose(in);";
        Printf.sprintf "  double *local = calloc(%Ld + 1, sizeof(double));"
          k.Loop_c.local_doubles;
        "  struct model_error err;";
        "  memset(&err, 0, sizeof err);";
        Printf.sprintf
          "  for (int w = 0; w < 4; w++) { %s; __asm__ volatile(\"\" ::: \
           \"memory\"); }"
          call;
        Printf.sprintf "  double ms[%d];" samples;
        Printf.sprintf "  for (int s = 0; s < %d; s++) {" samples;
        "    double t0 = now_ms();";
        Printf.sprintf
          "    for (int r = 0; r < %d; r++) { %s; __asm__ volatile(\"\" ::: \
           \"memory\"); }"
          b.Bench.reps call;
        "    ms[s] = now_ms() - t0;";
        "  }";
        "  int rc = (int)err.kind;";
        "  FILE *out = fopen(argv[2], \"wb\");";
        "  if (!out) return 4;";
        "  fwrite(&rc, sizeof rc, 1, out);";
        "  fwrite(ms, sizeof ms[0], sizeof ms / sizeof ms[0], out);";
        "  fwrite(blob, 1, total, out);";
        "  fclose(out);";
        "  return 0;";
        "}";
        "";
      ]
  in
  String.concat "\n"
    [
      "#define _POSIX_C_SOURCE 200809L";
      Loop_c_runtime.prelude;
      Loop_c_runtime.helpers k.Loop_c.helpers;
      k.Loop_c.source;
      main;
    ]

type run = {
  precision : Loop_numerics.Precision.t;
  source_bytes : int;
  samples_ms : float array;
  outputs : Tensor.packed Tensor_id.Map.t;
}

let ( let* ) = Result.bind

let execute_c (b : Bench.t) (c : Config.t) =
  let p = b.Bench.program in
  let* k =
    Result.map_error
      (fun e -> Fmt.str "%s: %a" c.Config.name Loop_c.pp_error e)
      (Err.payload
         (Loop_c.kernel ?vector:c.Config.vector ~numerics:c.Config.numerics
            ?precision:c.Config.precision ~name:"kernel" p))
  in
  let text = timed_source b k in
  (match Sys.getenv_opt "BENCH_DUMP" with
  | Some d ->
      Loop_c_exec.Proc.write_file
        (Filename.concat d
           (String.map (fun c -> if c = ' ' then '_' else c) b.Bench.name
           ^ "." ^ c.Config.name ^ ".c"))
        text
  | None -> ());
  let* exe =
    Result.map_error
      (function `C_compile m | `C_host m -> m)
      (Loop_c_exec.compile text)
  in
  let* tensors =
    Result.map_error
      (fun _ -> "binding failed")
      (Loop_c_exec.bind_buffers ~outputs:(fun _ -> None) p ~bind:(bind p))
  in
  let offsets, total = Loop_c_exec.layout p in
  let dir = Loop_c_exec.Proc.temp_dir "loop_fp32_bench" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () ->
      let blob = Filename.concat dir "in.bin"
      and result = Filename.concat dir "out.bin" in
      let fd = Unix.openfile blob [ Unix.O_RDWR; Unix.O_CREAT ] 0o600 in
      Unix.ftruncate fd (max total 1);
      List.iter2
        (fun (buf : Loop_buffer.t) off ->
          if buf.Loop_buffer.role = Loop_buffer.Input then
            Loop_c_exec.Blob.copy `In fd off
              (Tensor_id.Map.find buf.Loop_buffer.id tensors))
        p.Loop_program.buffers offsets;
      Unix.close fd;
      let* st, log = Loop_c_exec.Proc.run [ exe; blob; result ] in
      if st <> Loop_c_exec.Proc.Exited 0 then Error ("run failed: " ^ log)
      else
        let s = Loop_c_exec.Proc.read_file result in
        let header = 4 + (8 * samples) in
        if Int32.to_int (String.get_int32_le s 0) <> 0 then
          Error "the kernel reported a failure"
        else
          let samples_ms =
            Array.init samples (fun i ->
                Int64.float_of_bits (String.get_int64_le s (4 + (i * 8))))
          in
          let fd = Unix.openfile result [ Unix.O_RDWR ] 0o600 in
          List.iter2
            (fun (buf : Loop_buffer.t) off ->
              if buf.Loop_buffer.role = Loop_buffer.Output then
                Loop_c_exec.Blob.copy `Out fd (header + off)
                  (Tensor_id.Map.find buf.Loop_buffer.id tensors))
            p.Loop_program.buffers offsets;
          Unix.close fd;
          let outputs =
            List.fold_left
              (fun acc (buf : Loop_buffer.t) ->
                if buf.Loop_buffer.role = Loop_buffer.Output then
                  Tensor_id.Map.add buf.Loop_buffer.id
                    (Tensor_id.Map.find buf.Loop_buffer.id tensors)
                    acc
                else acc)
              Tensor_id.Map.empty p.Loop_program.buffers
          in
          Ok
            {
              precision = k.Loop_c.precision;
              source_bytes = String.length text;
              samples_ms;
              outputs;
            })

(* ---- the Wasm route -------------------------------------------------------- *)

(* Instantiates the module once, warms it for 100 ms (so V8 has tiered up past its
   baseline compiler), then times [reps] calls nine times. Writes
   [status, samples, blob]. *)
let wasm_runner =
  {|const fs = require("fs");
const [wasmPath, inPath, outPath, heapBase, total, localBase, reps, ...offsets] = process.argv.slice(2);
const inst = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)),
  { math: { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos } });
const mem = inst.exports.memory.buffer;
const base = Number(heapBase), size = Number(total), local = Number(localBase), n = Number(reps);
new Uint8Array(mem).fill(0xAB, local, base);
new Uint8Array(mem).set(fs.readFileSync(inPath), base);
const ptrs = offsets.map((o) => base + Number(o));
const call = () => inst.exports.loop_kernel(local, ...ptrs);
let rc = call();
const t0 = performance.now();
while (performance.now() - t0 < 100) for (let i = 0; i < 4; i++) rc = rc || call();
const samples = [];
for (let s = 0; s < 9; s++) {
  const t = performance.now();
  for (let r = 0; r < n; r++) rc = rc || call();
  samples.push(performance.now() - t);
}
const out = Buffer.alloc(4 + 8 * 9 + size);
out.writeInt32LE(rc, 0);
samples.forEach((x, i) => out.writeDoubleLE(x, 4 + 8 * i));
out.set(new Uint8Array(mem, base, size), 4 + 8 * 9);
fs.writeFileSync(outPath, out);
|}

let execute_wasm (b : Bench.t) (c : Config.t) =
  let p = b.Bench.program in
  let* lowered =
    Result.map_error
      (fun e -> Fmt.str "%s: %a" c.Config.name Loop_wasm.pp_error e)
      (Err.payload
         (Loop_wasm.lower ?vector:c.Config.vector ~numerics:c.Config.numerics
            ?precision:c.Config.precision p))
  in
  let* tensors =
    Result.map_error
      (fun _ -> "binding failed")
      (Loop_c_exec.bind_buffers ~outputs:(fun _ -> None) p ~bind:(bind p))
  in
  let offsets, total = Loop_c_exec.layout p in
  let heap_base = lowered.Loop_wasm.heap_base in
  let pages = max 1 ((heap_base + total + 65535) / 65536) in
  let* wasm =
    Result.map_error
      (fun (`Wasm_invalid i) ->
        Fmt.str "%a" Wasm_check.pp_error (`Wasm_invalid i))
      (Err.payload (Wasm_encode.module_ (Loop_wasm.with_pages lowered ~pages)))
  in
  let features = Wasm_features.of_module lowered.Loop_wasm.module_ in
  let dir = Loop_c_exec.Proc.temp_dir "loop_fp32_bench_wasm" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () ->
      let module_file = Filename.concat dir "k.wasm"
      and blob = Filename.concat dir "in.bin"
      and result = Filename.concat dir "out.bin"
      and runner = Filename.concat dir "runner.js" in
      Loop_c_exec.Proc.write_file module_file wasm;
      Loop_c_exec.Proc.write_file runner wasm_runner;
      let fd = Unix.openfile blob [ Unix.O_RDWR; Unix.O_CREAT ] 0o600 in
      Unix.ftruncate fd (max total 1);
      List.iter2
        (fun (buf : Loop_buffer.t) off ->
          if buf.Loop_buffer.role = Loop_buffer.Input then
            Loop_c_exec.Blob.copy `In fd off
              (Tensor_id.Map.find buf.Loop_buffer.id tensors))
        p.Loop_program.buffers offsets;
      Unix.close fd;
      let argv =
        Loop_wasm_exec.Node.command features
        @ [
            runner;
            module_file;
            blob;
            result;
            string_of_int heap_base;
            string_of_int total;
            string_of_int lowered.Loop_wasm.local_base;
            string_of_int b.Bench.reps;
          ]
        @ List.map string_of_int offsets
      in
      let* st, log = Loop_c_exec.Proc.run argv in
      if st <> Loop_c_exec.Proc.Exited 0 then Error ("node failed: " ^ log)
      else
        let s = Loop_c_exec.Proc.read_file result in
        if Int32.to_int (String.get_int32_le s 0) <> 0 then
          Error "the kernel reported a failure"
        else
          let samples_ms =
            Array.init samples (fun i ->
                Int64.float_of_bits (String.get_int64_le s (4 + (i * 8))))
          in
          let header = 4 + (8 * samples) in
          let fd = Unix.openfile result [ Unix.O_RDWR ] 0o600 in
          List.iter2
            (fun (buf : Loop_buffer.t) off ->
              if buf.Loop_buffer.role = Loop_buffer.Output then
                Loop_c_exec.Blob.copy `Out fd (header + off)
                  (Tensor_id.Map.find buf.Loop_buffer.id tensors))
            p.Loop_program.buffers offsets;
          Unix.close fd;
          let outputs =
            List.fold_left
              (fun acc (buf : Loop_buffer.t) ->
                if buf.Loop_buffer.role = Loop_buffer.Output then
                  Tensor_id.Map.add buf.Loop_buffer.id
                    (Tensor_id.Map.find buf.Loop_buffer.id tensors)
                    acc
                else acc)
              Tensor_id.Map.empty p.Loop_program.buffers
          in
          Ok
            {
              precision = lowered.Loop_wasm.precision;
              source_bytes = String.length wasm;
              samples_ms;
              outputs;
            })

let execute b c =
  match backend with `C -> execute_c b c | `Wasm -> execute_wasm b c

(* ---- verification (independent of timing) -------------------------------- *)

let interp ?precision ?fused p =
  match Err.payload (Loop_interp.run ?precision ?fused p ~bind:(bind p)) with
  | Ok m -> m
  | Error e -> Fmt.failwith "interpreter: %a" Loop_interp.pp_error e

let equal_outputs = Tensor_id.Map.equal Loop_check.tensors_equal

(* Largest absolute and relative difference between two result sets. *)
let diff a b =
  Tensor_id.Map.fold
    (fun id x (abs_, rel) ->
      let y = Tensor_id.Map.find id b in
      let (Tensor.Tensor t) = x in
      let abs_ = ref abs_ and rel = ref rel in
      Vec6.iter t.Tensor.shape (fun c ->
          let u = Tensor.read_at x (Vec6.get c)
          and w = Tensor.read_at y (Vec6.get c) in
          let d = Float.abs (u -. w) in
          abs_ := Float.max !abs_ d;
          if w <> 0. then rel := Float.max !rel (d /. Float.abs w));
      (!abs_, !rel))
    a (0., 0.)

(* The one cell of the first output nudged by one ulp of binary32. *)
let corrupt outputs =
  let id, t = Tensor_id.Map.choose outputs in
  let (Tensor.Tensor tt) = t in
  let copy =
    Tensor.materialize tt.Tensor.shape (fun c -> Tensor.read_at t (Vec6.get c))
  in
  let c0 = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
  let bits = Int32.bits_of_float (Tensor.read copy c0) in
  Tensor.set_float copy c0 (Int32.float_of_bits (Int32.succ bits));
  Tensor_id.Map.add id copy outputs

let median a =
  let s = Array.copy a in
  Array.sort compare s;
  s.(Array.length s / 2)

let json_floats a =
  "["
  ^ String.concat "," (Array.to_list (Array.map (Printf.sprintf "%.4f") a))
  ^ "]"

let compiler_text () =
  match backend with
  | `C -> String.concat " " !Loop_c_exec.compiler
  | `Wasm -> (
      match Loop_c_exec.Proc.run (!Loop_wasm_exec.node @ [ "--version" ]) with
      | Ok (_, v) -> "node " ^ String.trim v
      | Error _ -> "node")

let () =
  let selftest = Array.exists (String.equal "--selftest") Sys.argv in
  let failures = ref 0 in
  let results = ref [] in
  Printf.eprintf "%-22s %-18s %-5s %10s %10s %9s %10s\n%!" "program" "config"
    "prec" "median ms" "per call" "vs f64" "max |err|";
  List.iter
    (fun (b : Bench.t) ->
      let p = b.Bench.program in
      let f64_ref = interp p in
      let f64_median = ref nan in
      List.iter
        (fun (c : Config.t) ->
          match execute b c with
          | Error m ->
              incr failures;
              Printf.eprintf "%-22s %-18s FAILED: %s\n%!" b.Bench.name
                c.Config.name m
          | Ok r ->
              (* The answer a configuration is defined to give: its plan's own
                 scalar program (scheduled sums spelled out) at its precision. *)
              let plan =
                match c.Config.precision with
                | Some precision -> (
                    match
                      Loop_plan.force ?target:c.Config.vector ~precision p
                    with
                    | Ok plan -> plan
                    | Error _ -> failwith "forced precision refused")
                | None ->
                    Loop_plan.resolve ?target:c.Config.vector
                      ~numerics:c.Config.numerics p
              in
              let expected =
                interp ~precision:plan.Loop_plan.precision
                  (Loop_plan.oracle plan p)
              in
              (* A relaxed multiply-add is the fused or the unfused answer at the
                 engine's choice: either whole-kernel oracle is admissible. *)
              let relaxed_madd =
                match c.Config.vector with
                | Some t -> (Loop_target.f32 t).Loop_target.relaxed_madd
                | None -> false
              in
              let exact =
                equal_outputs expected r.outputs
                || relaxed_madd
                   && equal_outputs
                        (interp ~precision:plan.Loop_plan.precision ~fused:false
                           (Loop_plan.oracle plan p))
                        r.outputs
              in
              let rejects_corrupt =
                not (equal_outputs expected (corrupt r.outputs))
              in
              if not exact then incr failures;
              if selftest && not rejects_corrupt then incr failures;
              let m = median r.samples_ms in
              if c.Config.name = "f64_scalar" then f64_median := m;
              let max_abs, max_rel = diff r.outputs f64_ref in
              Printf.eprintf
                "%-22s %-18s %-5s %10.3f %8.2f us %8.2fx %10.2e%s\n%!"
                b.Bench.name c.Config.name
                (Loop_numerics.Precision.name r.precision)
                m
                (m /. float_of_int b.Bench.reps *. 1000.)
                (!f64_median /. m) max_abs
                (if exact then "" else "  WRONG");
              results :=
                Printf.sprintf
                  "{\"backend\":%S,\"program\":%S,\"config\":%S,\"precision\":%S,\"reps\":%d,\"samples_ms\":%s,\"median_ms\":%.4f,\"exact_vs_oracle\":%b,\"rejects_corruption\":%b,\"max_abs_vs_f64\":%g,\"max_rel_vs_f64\":%g,\"source_bytes\":%d,\"compiler\":%S}"
                  (match backend with `C -> "c" | `Wasm -> "wasm")
                  b.Bench.name c.Config.name
                  (Loop_numerics.Precision.name r.precision)
                  b.Bench.reps (json_floats r.samples_ms) m exact
                  rejects_corrupt max_abs max_rel r.source_bytes
                  (compiler_text ())
                :: !results)
        (Config.all backend))
    benches;
  List.iter print_endline (List.rev !results);
  Array.iter
    (fun a ->
      let prefix = "--json=" in
      if String.starts_with ~prefix a then
        Loop_c_exec.Proc.write_file
          (String.sub a (String.length prefix)
             (String.length a - String.length prefix))
          (String.concat "\n" (List.rev !results) ^ "\n"))
    Sys.argv;
  if !failures > 0 then (
    Printf.eprintf "loop_fp32_bench: %d verification failure(s)\n%!" !failures;
    exit 1)
