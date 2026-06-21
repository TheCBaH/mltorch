(* Hermetic unit tests for the pure-OCaml pt2 reader. No network, no libtorch:
   we build a tiny STORED zip and a minimal tensor pickle in memory and check
   the reader round-trips them. *)

let le16 b v =
  Buffer.add_char b (Char.chr (v land 0xff));
  Buffer.add_char b (Char.chr ((v lsr 8) land 0xff))

let le32 b v =
  le16 b (v land 0xffff);
  le16 b ((v lsr 16) land 0xffff)

(* Build a ZIP archive from [(name, data)] entries (STORED) via zipc — the same
   writer our reader is built on, so it exercises a real, CRC-correct archive. *)
let make_zip entries =
  let add z (path, data) =
    let file = Result.get_ok (Zipc.File.stored_of_binary_string data) in
    let m = Result.get_ok (Zipc.Member.make ~path (Zipc.Member.File file)) in
    Zipc.add m z
  in
  Result.get_ok (Zipc.to_binary_string (List.fold_left add Zipc.empty entries))

let%expect_test "zip round-trips STORED entries (incl. relative prefix)" =
  let zip =
    Pt2_zip.of_string
      (make_zip
         [ ("m/models/model.json", "{\"k\":1}"); ("m/data/0", "raw-bytes") ])
  in
  Printf.printf "prefix=%s\n" (Pt2_zip.prefix zip);
  Printf.printf "model.json=%s\n" (Pt2_zip.read_rel_exn zip "models/model.json");
  Printf.printf "data/0=%s\n" (Pt2_zip.read_rel_exn zip "data/0");
  Printf.printf "missing=%b\n" (Pt2_zip.read zip "m/nope" = None);
  [%expect
    {|
    prefix=m
    model.json={"k":1}
    data/0=raw-bytes
    missing=true
    |}]

(* --- Pt2_tensor --- *)

let%expect_test "metadata: numel, contiguity, pp" =
  let mk strides =
    {
      Pt2_tensor.dtype = Pt2_dtype.Float32;
      sizes = [ 2; 3 ];
      strides;
      storage_offset = 0;
      data = Bytes.create (6 * 4);
    }
  in
  let c = mk [ 3; 1 ]
  (* row-major *)
  and s =
    mk [ 1; 2 ]
    (* column-major *)
  in
  Format.printf "%a numel=%d contig(c)=%b contig(s)=%b@." Pt2_tensor.pp c
    (Pt2_tensor.numel c)
    (Pt2_tensor.is_contiguous c)
    (Pt2_tensor.is_contiguous s);
  [%expect {| float32[2; 3] numel=6 contig(c)=true contig(s)=false |}]

(* --- Pt2_pickle --- *)

(* Hand-encode the pickle torch.save emits for one tensor:
   _rebuild_tensor_v2((storage, FloatStorage, '0', 'cpu', 4), 0, (2,2), (2,1),
   False, OrderedDict()). *)
let tensor_pickle () =
  let b = Buffer.create 128 in
  let op c = Buffer.add_char b (Char.chr c) in
  let global m n =
    op 0x63;
    Buffer.add_string b (m ^ "\n" ^ n ^ "\n")
  in
  let unicode s =
    op 0x58;
    le32 b (String.length s);
    Buffer.add_string b s
  in
  let int1 v =
    op 0x4b;
    Buffer.add_char b (Char.chr v)
  in
  op 0x80;
  Buffer.add_char b '\002' (* PROTO 2 *);
  global "torch._utils" "_rebuild_tensor_v2";
  op 0x28 (* MARK: start args *);
  op 0x28 (* MARK: start persid tuple *);
  unicode "storage";
  global "torch" "FloatStorage";
  unicode "0";
  unicode "cpu";
  int1 4;
  op 0x74 (* TUPLE -> persid *);
  op 0x51 (* BINPERSID *);
  int1 0 (* storage_offset *);
  op 0x28;
  int1 2;
  int1 2;
  op 0x74 (* size tuple (2,2) *);
  op 0x28;
  int1 2;
  int1 1;
  op 0x74 (* stride tuple (2,1) *);
  op 0x89 (* NEWFALSE: requires_grad *);
  op 0x7d (* EMPTY_DICT: hooks *);
  op 0x74 (* TUPLE -> args *);
  op 0x52 (* REDUCE *);
  op 0x2e (* STOP *);
  Buffer.contents b

let%expect_test "pickle yields rebuild descriptor" =
  let rb = Pt2_pickle.parse_tensor (tensor_pickle ()) in
  Printf.printf "key=%s dtype=%s offset=%d sizes=[%s] strides=[%s]\n"
    rb.Pt2_pickle.storage_key
    (Pt2_dtype.to_string rb.Pt2_pickle.dtype)
    rb.Pt2_pickle.storage_offset
    (String.concat ";" (List.map string_of_int rb.Pt2_pickle.sizes))
    (String.concat ";" (List.map string_of_int rb.Pt2_pickle.strides));
  [%expect {| key=0 dtype=float32 offset=0 sizes=[2;2] strides=[2;1] |}]
