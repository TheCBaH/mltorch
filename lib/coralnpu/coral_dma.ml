open Err.Syntax

let base = 0x4005_0000L

module Register = struct
  type t = Ctrl | Cur_desc | Desc_addr | Status | Xfer_remain

  let offset = function
    | Ctrl -> 0x00
    | Status -> 0x04
    | Desc_addr -> 0x08
    | Cur_desc -> 0x0C
    | Xfer_remain -> 0x10
end

let ctrl_start = 3
let status_done = 2
let status_error = 4

module Beat = struct
  type t = B1 | B2 | B4 | B8 | B16

  let log2 = function B1 -> 0 | B2 -> 1 | B4 -> 2 | B8 -> 3 | B16 -> 4
  let bytes b = 1 lsl log2 b

  let of_bytes = function
    | 1 -> Some B1
    | 2 -> Some B2
    | 4 -> Some B4
    | 8 -> Some B8
    | 16 -> Some B16
    | _ -> None
end

module Descriptor = struct
  type t = {
    beat : Beat.t;
    dst : int64;
    dst_fixed : bool;
    length : int64;
    src : int64;
    src_fixed : bool;
  }

  let size = 32

  let copy ?(beat = Beat.B16) ~src ~dst length =
    { beat; dst; dst_fixed = false; length; src; src_fixed = false }
end

let max_length = 0xFF_FFFFL

let len_flags (d : Descriptor.t) =
  let bit b n = if b then 1 lsl n else 0 in
  Int32.of_int
    (Int64.to_int d.length
    lor (Beat.log2 d.beat lsl 24)
    lor bit d.src_fixed 27 lor bit d.dst_fixed 28)

module Misaligned = struct
  type t = { address : int64; alignment : int }
end

module Unreachable = struct
  type t = { address : int64; length : int64 }
end

type error =
  [ `Dma_length of int64
  | `Dma_misaligned of Misaligned.t
  | `Dma_unreachable of Unreachable.t ]

let pp_error ppf : error -> unit = function
  | `Dma_length n -> Fmt.pf ppf "a DMA transfer of %Ld bytes" n
  | `Dma_misaligned { Misaligned.address; alignment } ->
      Fmt.pf ppf "DMA address 0x%08Lx is not %d-byte aligned" address alignment
  | `Dma_unreachable { Unreachable.address; length } ->
      Fmt.pf ppf "%Ld bytes at 0x%08Lx are outside every region the DMA reaches"
        length address

let aligned address alignment =
  if Int64.rem address (Int64.of_int alignment) = 0L then Err.return ()
  else Err.fail (`Dma_misaligned { Misaligned.address; alignment })

let reachable (p : Coral_profile.t) ~address ~length =
  if
    List.exists
      (fun (r : Coral_profile.Region.t) ->
        r.dma_reachable && address >= r.base
        && Int64.add address length <= Int64.add r.base r.size)
      p.regions
  then Err.return ()
  else Err.fail (`Dma_unreachable { Unreachable.address; length })

let check p (d : Descriptor.t) =
  let beat = Beat.bytes d.beat in
  let* () =
    if
      d.length <= 0L || d.length > max_length
      || Int64.rem d.length (Int64.of_int beat) <> 0L
    then Err.fail (`Dma_length d.length)
    else Err.return ()
  in
  let* () = aligned d.src beat in
  let* () = aligned d.dst beat in
  let span fixed = if fixed then Int64.of_int beat else d.length in
  let* () = reachable p ~address:d.src ~length:(span d.src_fixed) in
  reachable p ~address:d.dst ~length:(span d.dst_fixed)

let chain p ~at descriptors =
  let n = List.length descriptors in
  let* () = aligned at Descriptor.size in
  let* () =
    reachable p ~address:at ~length:(Int64.of_int (n * Descriptor.size))
  in
  let b = Bytes.make (n * Descriptor.size) '\000' in
  let* () =
    Err.List.iter
      (fun (k, (d : Descriptor.t)) ->
        let* () = check p d in
        let o = k * Descriptor.size in
        let next =
          if k = n - 1 then 0l
          else
            Int64.to_int32
              (Int64.add at (Int64.of_int ((k + 1) * Descriptor.size)))
        in
        Bytes.set_int32_le b o (Int64.to_int32 d.src);
        Bytes.set_int32_le b (o + 4) (Int64.to_int32 d.dst);
        Bytes.set_int32_le b (o + 8) (len_flags d);
        Bytes.set_int32_le b (o + 12) next;
        Err.return ())
      (List.mapi (fun k d -> (k, d)) descriptors)
  in
  Err.return (Bytes.to_string b)
