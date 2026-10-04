open Loop_ir

type error =
  [ Loop_interp.error
  | `C_compile of string
  | `C_host of string
  | `C_unsupported of Loop_c.error ]

let pp_error ppf : [< error ] -> unit = function
  | `C_compile m -> Fmt.pf ppf "the C compiler rejected the source: %s" m
  | `C_host m -> Fmt.pf ppf "the C host failed: %s" m
  | `C_unsupported e -> Loop_c.pp_error ppf e
  | #Loop_interp.error as e -> Loop_interp.pp_error ppf e

let ( let* ) = Result.bind

module Proc = C_proc
module Host = C_host

let compiler = ref C_host.default_compiler
let align = 64
let round_up n = (n + align - 1) / align * align

let cell_bytes (b : Loop_buffer.t) =
  Payload.packed_cell_bytes b.Loop_buffer.sg.Tensor_sig.fmt

let numel (b : Loop_buffer.t) =
  Int64.to_int
    (Int64.of_int (Dim.to_int (Vec6.numel b.Loop_buffer.sg.Tensor_sig.shape)))

(* Byte offset of each buffer in the blob, and the blob's size. *)
let layout (p : Loop_program.t) =
  let offsets, total =
    List.fold_left
      (fun (acc, off) b ->
        (off :: acc, off + round_up (numel b * cell_bytes b)))
      ([], 0) p.Loop_program.buffers
  in
  (List.rev offsets, total)

let kernel_name = "kernel"

let unit_text ?(dialect = Loop_c_dialect.Gnu) ?vector ?numerics ?precision
    ?fuse_reductions p =
  match
    Err.payload
      (Loop_c.kernel ~dialect ?vector ?numerics ?precision ?fuse_reductions
         ~name:kernel_name p)
  with
  | Error e -> Error e
  | Ok k ->
      Ok
        ( k,
          String.concat "\n"
            [
              Loop_c_runtime.prelude_in dialect;
              Loop_c_runtime.helpers ~dialect k.Loop_c.helpers;
              k.Loop_c.source;
            ] )

let source ?dialect ?vector ?numerics ?precision ?fuse_reductions p =
  match unit_text ?dialect ?vector ?numerics ?precision ?fuse_reductions p with
  | Error e -> Err.fail (`C_unsupported e)
  | Ok (k, unit) ->
      let offsets, total = layout p in
      let args =
        List.map2
          (fun ty off -> Printf.sprintf "(%s *)(blob + %d)" ty off)
          k.Loop_c.buffer_types offsets
      in
      let main =
        String.concat "\n"
          [
            "#include <stdio.h>";
            "#include <stdlib.h>";
            "int main(int argc, char **argv) {";
            "  if (argc != 3) return 2;";
            Printf.sprintf "  const size_t total = %d;" total;
            "  unsigned char *blob = malloc(total ? total : 1);";
            "  FILE *in = fopen(argv[1], \"rb\");";
            "  if (!blob || !in || fread(blob, 1, total, in) != total) return \
             3;";
            "  fclose(in);";
            Printf.sprintf "  double *local = calloc(%Ld + 1, sizeof(double));"
              k.Loop_c.local_doubles;
            "  struct model_error err;";
            "  memset(&err, 0, sizeof err);";
            Printf.sprintf "  int rc = %s(&err, local%s);" kernel_name
              (String.concat "" (List.map (fun a -> ", " ^ a) args));
            "  FILE *out = fopen(argv[2], \"wb\");";
            "  if (!out) return 4;";
            "  int32_t rc32 = rc;";
            "  fwrite(&rc32, sizeof rc32, 1, out);";
            "  fwrite(&err.kind, sizeof err.kind, 1, out);";
            Printf.sprintf "  fwrite(err.v, sizeof err.v[0], %d, out);"
              Loop_c_runtime.error_words;
            "  fwrite(blob, 1, total, out);";
            "  fclose(out);";
            "  free(blob); free(local);";
            "  return 0;";
            "}";
            "";
          ]
      in
      let text = String.concat "\n" [ unit; main ] in
      Err.return (text, List.map string_of_int offsets)

module Blob = C_blob
module Payload_io = C_payload_io

let memo : (string, string) Hashtbl.t = Hashtbl.create 16

let build_dir =
  lazy
    (let dir = C_proc.temp_dir "loop_c_exec" in
     at_exit (fun () -> C_proc.remove_tree dir);
     dir)

let compile text =
  let key =
    Digest.to_hex (Digest.string (String.concat " " !compiler ^ text))
  in
  match Hashtbl.find_opt memo key with
  | Some exe -> Ok exe
  | None -> (
      let dir = Lazy.force build_dir in
      let c = Filename.concat dir (key ^ ".c") in
      let exe = Filename.concat dir key in
      C_proc.write_file c text;
      match C_proc.run (!compiler @ [ c; "-o"; exe; "-lm" ]) with
      | Error m -> Error (`C_host m)
      | Ok (C_proc.Exited 0, _) ->
          Hashtbl.add memo key exe;
          Ok exe
      | Ok (st, log) ->
          Error
            (`C_compile
               (Format.asprintf "%a@.%s@.source: %s" C_proc.pp_status st log c))
      )

let word_bytes = 8
let read_i32 s pos = Int32.to_int (String.get_int32_le s pos)

let check_and_add acc (b : Loop_buffer.t) tensor =
  match
    Err.payload
      (Kernel_eval.check_binding b.Loop_buffer.id b.Loop_buffer.sg tensor)
  with
  | Error (`Binding_mismatch m) -> Error (`Binding_mismatch m)
  | Ok () -> Ok (Tensor_id.Map.add b.Loop_buffer.id tensor acc)

(* [Loop_interp]'s binding rules: an Input must be bound and match its
   signature, an Output is fresh (or a bound tensor, zero-filled), Scratch is
   always fresh. *)
let bind_buffers ~outputs (p : Loop_program.t) ~bind =
  List.fold_left
    (fun acc (b : Loop_buffer.t) ->
      let* acc = acc in
      match b.Loop_buffer.role with
      | Loop_buffer.Input -> (
          match bind b.Loop_buffer.id with
          | None -> Error (`Unbound_input b.Loop_buffer.id)
          | Some tensor -> check_and_add acc b tensor)
      | Loop_buffer.Output -> (
          match outputs b.Loop_buffer.id with
          | None ->
              Ok
                (Tensor_id.Map.add b.Loop_buffer.id (Loop_interp.allocate b) acc)
          | Some tensor ->
              let* acc = check_and_add acc b tensor in
              Tensor.zero_fill tensor;
              Ok acc)
      | Loop_buffer.Scratch ->
          Ok (Tensor_id.Map.add b.Loop_buffer.id (Loop_interp.allocate b) acc))
    (Ok Tensor_id.Map.empty) p.Loop_program.buffers

let exec ?dialect ?vector ?numerics ?precision ?fuse_reductions
    ?(outputs = fun _ -> None) (p : Loop_program.t) ~bind =
  let result : (Tensor.packed Tensor_id.Map.t, error) result =
    let* text, _ =
      Result.map_error
        (fun e -> (e : [ `C_unsupported of Loop_c.error ] :> error))
        (Err.payload
           (source ?dialect ?vector ?numerics ?precision ?fuse_reductions p))
    in
    let* exe = (compile text :> (string, error) result) in
    let* tensors =
      (bind_buffers ~outputs p ~bind
        :> (Tensor.packed Tensor_id.Map.t, error) result)
    in
    let offsets, total = layout p in
    let dir = Lazy.force build_dir in
    let blob = Filename.temp_file ~temp_dir:dir "in" ".bin" in
    let result_file = Filename.temp_file ~temp_dir:dir "out" ".bin" in
    Fun.protect
      ~finally:(fun () ->
        List.iter
          (fun f -> try Sys.remove f with Sys_error _ -> ())
          [ blob; result_file ])
      (fun () ->
        let fd = Unix.openfile blob [ Unix.O_RDWR ] 0o600 in
        Fun.protect
          ~finally:(fun () -> Unix.close fd)
          (fun () ->
            Unix.ftruncate fd (max total 1);
            List.iter2
              (fun (b : Loop_buffer.t) off ->
                match b.Loop_buffer.role with
                | Loop_buffer.Input ->
                    Blob.copy `In fd off
                      (Tensor_id.Map.find b.Loop_buffer.id tensors)
                | Loop_buffer.Output | Loop_buffer.Scratch -> ())
              p.Loop_program.buffers offsets);
        match C_proc.run [ exe; blob; result_file ] with
        | Error m -> Error (`C_host m)
        | Ok (((C_proc.Signaled _ | C_proc.Exited _) as st), log)
          when st <> C_proc.Exited 0 ->
            Error (`C_host (Format.asprintf "%a: %s" C_proc.pp_status st log))
        | Ok (_, _) ->
            let s = C_proc.read_file result_file in
            let header = 4 + 4 + (Loop_c_runtime.error_words * word_bytes) in
            (* Two int32 (status, kind) then the words; the blob follows. *)
            let header = header in
            if String.length s < header then Error (`C_host "short result file")
            else
              let rc = read_i32 s 0 in
              if rc <> 0 then
                let kind = read_i32 s 4 in
                let v =
                  Array.init Loop_c_runtime.error_words (fun i ->
                      String.get_int64_le s (8 + (i * word_bytes)))
                in
                match
                  Loop_c_failure.decode ~sites:(Loop_js_failure.sites p) ~kind
                    ~v
                with
                | Ok row -> Error (row :> error)
                | Error m -> Error (`C_host m)
              else
                let fd = Unix.openfile result_file [ Unix.O_RDWR ] 0o600 in
                Fun.protect
                  ~finally:(fun () -> Unix.close fd)
                  (fun () ->
                    List.iter2
                      (fun (b : Loop_buffer.t) off ->
                        match b.Loop_buffer.role with
                        | Loop_buffer.Output ->
                            Blob.copy `Out fd (header + off)
                              (Tensor_id.Map.find b.Loop_buffer.id tensors)
                        | Loop_buffer.Input | Loop_buffer.Scratch -> ())
                      p.Loop_program.buffers offsets);
                Ok
                  (List.fold_left
                     (fun acc (b : Loop_buffer.t) ->
                       match b.Loop_buffer.role with
                       | Loop_buffer.Output ->
                           Tensor_id.Map.add b.Loop_buffer.id
                             (Tensor_id.Map.find b.Loop_buffer.id tensors)
                             acc
                       | Loop_buffer.Input | Loop_buffer.Scratch -> acc)
                     Tensor_id.Map.empty p.Loop_program.buffers))
  in
  match result with Ok m -> Err.return m | Error e -> Err.fail e

(* A quantized buffer has no C implementation: the design admits none, and the
   emitter refuses it with a typed error. Here, and only here, a program refused
   for that reason is answered by the interpreter, so a shared fixture with a
   quantized operand stays comparable; any other refusal is a defect. *)
let executor_with ?dialect ?vector : Loop_check.Executor.t =
 fun p ~bind ->
  match Err.payload (exec ?dialect ?vector p ~bind) with
  | Error (`C_unsupported (`Unsupported_format (_, ("i16" | "i8")))) -> (
      match Err.payload (Loop_interp.run p ~bind) with
      | Ok m -> Err.return m
      | Error e -> Err.fail (e :> Loop_check.Executor.error))
  | Ok m -> Err.return m
  | Error (#Loop_interp.error as e) -> Err.fail (e :> Loop_check.Executor.error)
  | Error (`C_compile m) -> Err.fail (`Js_exception ("C compile: " ^ m))
  | Error (`C_host m) -> Err.fail (`Js_exception ("C host: " ^ m))
  | Error (`C_unsupported u) ->
      Err.fail (`Js_exception (Fmt.str "C unsupported: %a" Loop_c.pp_error u))

let executor = executor_with ?dialect:None ?vector:None

let executor_compcert =
  executor_with ~dialect:Loop_c_dialect.Compcert_scalar ?vector:None

let executor_vector = executor_with ?dialect:None ~vector:Loop_target.neon128
