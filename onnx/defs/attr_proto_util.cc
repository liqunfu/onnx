// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "attr_proto_util.h"

namespace ONNX_NAMESPACE {

#define ADD_BASIC_ATTR_IMPL(type, enumType, field)       \
  AttributeProto MakeAttribute(                          \
      const std::string& attr_name, const type& value) { \
    AttributeProto a;                                    \
    a.set_name(attr_name);                               \
    a.set_type(enumType);                                \
    a.set_##field(value);                                \
    return a;                                            \
  }

#define ADD_ATTR_IMPL(type, enumType, field)             \
  AttributeProto MakeAttribute(                          \
      const std::string& attr_name, const type& value) { \
    AttributeProto a;                                    \
    a.set_name(attr_name);                               \
    a.set_type(enumType);                                \
    *(a.mutable_##field()) = value;                      \
    return a;                                            \
  }

#define ADD_LIST_ATTR_IMPL(type, enumType, field)                      \
  AttributeProto MakeAttribute(                                        \
      const std::string& attr_name, const std::vector<type>& values) { \
    AttributeProto a;                                                  \
    a.set_name(attr_name);                                             \
    a.set_type(enumType);                                              \
    for (const auto& val : values) {                                   \
      *(a.mutable_##field()->Add()) = val;                             \
    }                                                                  \
    return a;                                                          \
  }

ADD_BASIC_ATTR_IMPL(float, AttributeProto_AttributeType_FLOAT, f)
ADD_BASIC_ATTR_IMPL(int64_t, AttributeProto_AttributeType_INT, i)
ADD_BASIC_ATTR_IMPL(std::string, AttributeProto_AttributeType_STRING, s)
ADD_ATTR_IMPL(TensorProto, AttributeProto_AttributeType_TENSOR, t)
ADD_LIST_ATTR_IMPL(float, AttributeProto_AttributeType_FLOATS, floats)
ADD_LIST_ATTR_IMPL(int64_t, AttributeProto_AttributeType_INTS, ints)
ADD_LIST_ATTR_IMPL(std::string, AttributeProto_AttributeType_STRINGS, strings)
ADD_LIST_ATTR_IMPL(TensorProto, AttributeProto_AttributeType_TENSORS, tensors)
ADD_LIST_ATTR_IMPL(GraphProto, AttributeProto_AttributeType_GRAPHS, graphs)

AttributeProto MakeRefAttribute(
    const std::string& attr_name,
    AttributeProto_AttributeType type) {
  AttributeProto a;
  a.set_name(attr_name);
  a.set_ref_attr_name(attr_name);
  a.set_type(type);
  return a;
}

AttributeProto MakeRefAttribute(
  const std::string& attr_name,
  const std::string& ref_attr_name,
  AttributeProto_AttributeType type) {
AttributeProto a;
a.set_name(attr_name);
a.set_ref_attr_name(ref_attr_name);
a.set_type(type);
return a;
}
} // namespace ONNX_NAMESPACE
