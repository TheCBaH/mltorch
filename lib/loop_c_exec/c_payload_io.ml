open Loop_ir
module P = C_payload_layout

let ( let* ) = Result.bind
let io f = try Ok (f ()) with Sys_error m | Failure m -> Error (`Io m)

let unix_io f =
  try Ok (f ()) with
  | Unix.Unix_error (e, fn, arg) ->
      Error (`Io (Printf.sprintf "%s(%s): %s" fn arg (Unix.error_message e)))
  | Sys_error m -> Error (`Io m)

let mkdir_p dir =
  let rec go d =
    if not (Sys.file_exists d) then (
      go (Filename.dirname d);
      Unix.mkdir d 0o755)
  in
  go dir

(* A payload file: the header, zero padding, and each entry's tensor at its
   offset. The file is created whole (zero-filled) and the tensors written into
   it, so padding is zero by construction. *)
let write_payload (layout : P.t) ~identity ~path tensors =
  unix_io (fun () ->
      let fd =
        Unix.openfile path [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_TRUNC ] 0o644
      in
      Fun.protect
        ~finally:(fun () -> Unix.close fd)
        (fun () ->
          Unix.ftruncate fd (Int64.to_int layout.P.length);
          let header = P.header layout ~identity in
          let (_ : int) =
            Unix.write_substring fd header 0 (String.length header)
          in
          List.iter2
            (fun (e : P.Entry.t) t ->
              C_blob.copy `In fd (Int64.to_int e.P.Entry.offset) t)
            layout.P.entries tensors))

let bound_tensors (layout : P.t) ~lookup ~missing =
  List.fold_left
    (fun acc (e : P.Entry.t) ->
      let* acc = acc in
      match lookup e.P.Entry.id with
      | None -> Error (missing e.P.Entry.id)
      | Some t -> (
          match
            Err.payload (Kernel_eval.check_binding e.P.Entry.id e.P.Entry.sg t)
          with
          | Error (`Binding_mismatch m) -> Error (`Binding_mismatch m)
          | Ok () -> Ok (t :: acc)))
    (Ok []) layout.P.entries
  |> Result.map List.rev

(* Validates an outputs file against [layout] and [identity], and decodes it. *)
let read_payload (layout : P.t) ~identity ~path =
  let expected = P.header layout ~identity in
  let* size =
    match (Unix.stat path).Unix.st_size with
    | n -> Ok n
    | exception Unix.Unix_error _ -> Error (`Bad_output ("cannot stat " ^ path))
  in
  if Int64.of_int size <> layout.P.length then
    Error
      (`Bad_output
         (Printf.sprintf "%d bytes, expected %Ld" size layout.P.length))
  else
    let* head =
      io (fun () ->
          let ic = open_in_bin path in
          Fun.protect
            ~finally:(fun () -> close_in_noerr ic)
            (fun () -> really_input_string ic 64))
    in
    if head <> expected then Error (`Bad_output "header mismatch")
    else
      unix_io (fun () ->
          let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
          Fun.protect
            ~finally:(fun () -> Unix.close fd)
            (fun () ->
              List.map
                (fun (e : P.Entry.t) ->
                  match Err.payload (Tensor.create_of_sig e.P.Entry.sg) with
                  | Error (`Quant_missing _) ->
                      raise (Failure "quantized output")
                  | Ok t ->
                      C_blob.copy `Out fd (Int64.to_int e.P.Entry.offset) t;
                      t)
                layout.P.entries))
