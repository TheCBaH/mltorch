open Loop_ir
module C = Loop_c_exec

type error =
  [ Loop_interp.error
  | `Wasm_host of string
  | `Wasm_invalid of Wasm_check.Invalid.t
  | `Wasm_unsupported of Loop_wasm.error ]

let pp_error ppf : [< error ] -> unit = function
  | `Wasm_host m -> Fmt.pf ppf "the Wasm host failed: %s" m
  | `Wasm_invalid i -> Wasm_check.pp_error ppf (`Wasm_invalid i)
  | `Wasm_unsupported e -> Loop_wasm.pp_error ppf e
  | #Loop_interp.error as e -> Loop_interp.pp_error ppf e

module Host = Wasm_host
module Via_c = Wasm_c_host
module Node = Wasm_node

let ( let* ) = Result.bind
let node = Wasm_host.node
let word_bytes = 8
let read_i32 s pos = Int32.to_int (String.get_int32_le s pos)

(* The runner reads the module and the input blob, places the blob at
   [heap_base] (the local region before it poisoned), calls [loop_kernel] with
   the local base and one pointer per buffer, and writes
   [status, kind, v[12], blob]: the C host's result file, so one decoder
   serves both. The math imports are the host's [Math]. *)
let runner =
  {|const fs = require("fs");
const [wasmPath, inPath, outPath, heapBase, total, localBase, markBase, ...offsets] = process.argv.slice(2);
const inst = new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync(wasmPath)),
  { math: { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos } });
const mem = inst.exports.memory.buffer;
const base = Number(heapBase), size = Number(total), local = Number(localBase);
// The local region starts dirty: a kernel zeroes what it uses.
new Uint8Array(mem).fill(0xAB, local, base);
new Uint8Array(mem).set(fs.readFileSync(inPath), base);
const rc = inst.exports.loop_kernel(local, ...offsets.map((o) => base + Number(o)));
const out = Buffer.alloc(104 + size + 24);
out.writeInt32LE(rc, 0);
out.writeInt32LE(new DataView(mem).getInt32(0, true), 4);
out.set(new Uint8Array(mem, 8, 96), 8);
out.set(new Uint8Array(mem, base, size), 104);
if (Number(markBase) >= 0) out.set(new Uint8Array(mem, Number(markBase), 24), 104 + size);
fs.writeFileSync(outPath, out);
|}

let build_dir =
  lazy
    (let dir = C.Proc.temp_dir "loop_wasm_exec" in
     at_exit (fun () -> C.Proc.remove_tree dir);
     dir)

let runner_path =
  lazy
    (let path = Filename.concat (Lazy.force build_dir) "runner.js" in
     C.Proc.write_file path runner;
     path)

let max_bytes = 0x8000_0000L

let exec_gen ~vector ~count_marks ?(outputs = fun _ -> None)
    (p : Loop_program.t) ~bind =
  let result : (Tensor.packed Tensor_id.Map.t * int list, error) result =
    let* lowered =
      Result.map_error
        (fun e -> `Wasm_unsupported e)
        (Err.payload (Loop_wasm.lower ?vector ~count_marks p))
    in
    let* tensors =
      (C.bind_buffers ~outputs p ~bind
        :> (Tensor.packed Tensor_id.Map.t, error) result)
    in
    let offsets, total = C.layout p in
    let heap_base = lowered.Loop_wasm.heap_base in
    let bytes = Int64.add (Int64.of_int heap_base) (Int64.of_int total) in
    if Int64.compare bytes max_bytes > 0 then
      Error (`Wasm_host "the buffers exceed the 2 GiB memory policy")
    else
      let pages =
        max 1 (Int64.to_int (Int64.div (Int64.add bytes 65535L) 65536L))
      in
      let m = Loop_wasm.with_pages lowered ~pages in
      let* wasm =
        Result.map_error
          (fun (`Wasm_invalid i) -> `Wasm_invalid i)
          (Err.payload (Wasm_encode.module_ m))
      in
      let dir = Lazy.force build_dir in
      let module_file = Filename.temp_file ~temp_dir:dir "k" ".wasm" in
      let blob = Filename.temp_file ~temp_dir:dir "in" ".bin" in
      let result_file = Filename.temp_file ~temp_dir:dir "out" ".bin" in
      Fun.protect
        ~finally:(fun () ->
          List.iter
            (fun f -> try Sys.remove f with Sys_error _ -> ())
            [ module_file; blob; result_file ])
        (fun () ->
          C.Proc.write_file module_file wasm;
          let fd = Unix.openfile blob [ Unix.O_RDWR ] 0o600 in
          Fun.protect
            ~finally:(fun () -> Unix.close fd)
            (fun () ->
              Unix.ftruncate fd (max total 1);
              List.iter2
                (fun (b : Loop_buffer.t) off ->
                  match b.Loop_buffer.role with
                  | Loop_buffer.Input ->
                      C.Blob.copy `In fd off
                        (Tensor_id.Map.find b.Loop_buffer.id tensors)
                  | Loop_buffer.Output | Loop_buffer.Scratch -> ())
                p.Loop_program.buffers offsets);
          let argv =
            !node
            @ [
                Lazy.force runner_path;
                module_file;
                blob;
                result_file;
                string_of_int heap_base;
                string_of_int total;
                string_of_int lowered.Loop_wasm.local_base;
                string_of_int
                  (Option.value lowered.Loop_wasm.mark_base ~default:(-1));
              ]
            @ List.map string_of_int offsets
          in
          match C.Proc.run argv with
          | Error m -> Error (`Wasm_host m)
          | Ok (st, log) when st <> C.Proc.Exited 0 ->
              Error
                (`Wasm_host (Format.asprintf "%a: %s" C.Proc.pp_status st log))
          | Ok (_, _) ->
              let s = C.Proc.read_file result_file in
              let header = 8 + (Loop_wasm_failure.error_words * word_bytes) in
              if String.length s < header then
                Error (`Wasm_host "short result file")
              else if read_i32 s 0 <> 0 then
                let kind = read_i32 s 4 in
                let v =
                  Array.init Loop_wasm_failure.error_words (fun i ->
                      String.get_int64_le s (8 + (i * word_bytes)))
                in
                match
                  Loop_c_failure.decode ~sites:lowered.Loop_wasm.sites ~kind ~v
                with
                | Ok row -> Error (row :> error)
                | Error m -> Error (`Wasm_host m)
              else
                let fd = Unix.openfile result_file [ Unix.O_RDWR ] 0o600 in
                Fun.protect
                  ~finally:(fun () -> Unix.close fd)
                  (fun () ->
                    List.iter2
                      (fun (b : Loop_buffer.t) off ->
                        match b.Loop_buffer.role with
                        | Loop_buffer.Output ->
                            C.Blob.copy `Out fd (header + off)
                              (Tensor_id.Map.find b.Loop_buffer.id tensors)
                        | Loop_buffer.Input | Loop_buffer.Scratch -> ())
                      p.Loop_program.buffers offsets);
                let counts =
                  if count_marks then
                    List.init (List.length Loop_mark.all) (fun k ->
                        read_i32 s (header + total + (4 * k)))
                  else []
                in
                Ok
                  ( List.fold_left
                      (fun acc (b : Loop_buffer.t) ->
                        match b.Loop_buffer.role with
                        | Loop_buffer.Output ->
                            Tensor_id.Map.add b.Loop_buffer.id
                              (Tensor_id.Map.find b.Loop_buffer.id tensors)
                              acc
                        | Loop_buffer.Input | Loop_buffer.Scratch -> acc)
                      Tensor_id.Map.empty p.Loop_program.buffers,
                    counts ))
  in
  match result with Ok m -> Err.return m | Error e -> Err.fail e

let exec ?vector ?outputs p ~bind =
  Err.map fst (exec_gen ~vector ~count_marks:false ?outputs p ~bind)

let exec_counted ?vector ?outputs p ~bind =
  Err.map
    (fun (m, counts) -> (m, List.combine Loop_mark.all counts))
    (exec_gen ~vector ~count_marks:true ?outputs p ~bind)

let executor_with ?vector : Loop_check.Executor.t =
 fun p ~bind ->
  match Err.payload (exec ?vector p ~bind) with
  | Ok m -> Err.return m
  | Error (#Loop_interp.error as e) -> Err.fail (e :> Loop_check.Executor.error)
  | Error (`Wasm_host m) -> Err.fail (`Js_exception ("Wasm host: " ^ m))
  | Error (`Wasm_invalid i) ->
      Err.fail
        (`Js_exception
           (Fmt.str "Wasm invalid: %a" Wasm_check.pp_error (`Wasm_invalid i)))
  | Error (`Wasm_unsupported u) ->
      Err.fail
        (`Js_exception (Fmt.str "Wasm unsupported: %a" Loop_wasm.pp_error u))

let executor = executor_with ?vector:None
let executor_simd = executor_with ~vector:Loop_target.wasm128
