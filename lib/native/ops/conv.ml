(* Convolution ops. This file is now a thin facade: Conv1d (which depends on
   Conv2d) lives in conv_conv1d.ml, Conv2d lives in conv_conv2d.ml,
   Conv2d_padding (which depends on Conv2d) in conv_padding.ml, Convolution
   (which also depends on Conv2d) in conv_convolution.ml, and Conv3d (its own
   Compute, not a Conv2d delegation) in conv_conv3d.ml. Every alias below is
   exactly what was a top-level module in this file before the split, so
   external references such as [Conv.Conv2d] or [Conv.Convolution] are
   unaffected.. *)

module Conv1d = Conv_conv1d.Conv1d
module Conv2d = Conv_conv2d.Conv2d
module Conv2d_padding = Conv_padding.Conv2d_padding
module Conv3d = Conv_conv3d.Conv3d
module Convolution = Conv_convolution.Convolution
