from tbe import tik
import numpy as np
from tbe.common.platform import set_current_compile_soc_info
import tbe.common.platform as tbe_platform


def element_add_test():
    tik_instance = tik.Tik()
    set_current_compile_soc_info("Ascend310")
    data_A = tik_instance.Tensor("float16", (1024, 256), name="data_A", scope=tik.scope_gm)
    data_B = tik_instance.Tensor("float16", (1024, 256), name="data_B", scope=tik.scope_gm)

    # If you want to achieve multi-core, you need to first obtain the number of cores
    # and set the calculation range for each core
    input_num = 256 * 1024
    aicore_num = tbe_platform.get_soc_spec("CORE_NUM")  # aicore_num is 2
    data_num_each_core = input_num // aicore_num

    data_a_relu = tik_instance.Tensor("float16", (128, 128), name="data_a_relu", scope=tik.scope_ubuf)
    data_a_ub = tik_instance.Tensor("float16", (128, 128), name="data_a_ub", scope=tik.scope_ubuf)
    data_a_exp = tik_instance.Tensor("float16", (128, 128), name="data_a_exp", scope=tik.scope_ubuf)
    with tik_instance.for_range(0, 4) as i:
        with tik_instance.for_range(0, 2) as j:
            offset = 256*256*i+128*j+256
            tik_instance.data_move(data_a_ub, data_A[offset], 0, 128, 128//16, 128//16*3, 0)
            tik_instance.vec_relu(128, data_a_relu, data_a_ub, 128, 8, 8)
            tik_instance.data_move(data_B[offset], data_a_relu, 0, 128, 128//16, 0, 128//16*3)
    with tik_instance.for_range(0, 4) as i:
        with tik_instance.for_range(0, 2) as j:
            offset = 256*256*i+128*j
            tik_instance.data_move(data_a_ub, data_A[offset], 0, 128, 128//16, 128//16*3, 0)
            tik_instance.vec_exp(128, data_a_exp, data_a_ub, 128, 8, 8)
    # move data from out to UB, then calculate with TIK API and move data from UB to OUT.
    # making full use of the "stride" parameter in the API to reduce the number of loops.

    tik_instance.BuildCCE(kernel_name="element_add_test",inputs=[data_A], outputs=[data_B])

    return tik_instance


def compareData(src, exp, print_n):
    # pass
    totolNum = src.reshape(-1).shape[0]
    errorCnt = 0
    for i in range(totolNum):
        if (abs(src[i] - exp[i])) / abs(exp[i]) > 0.01:
            if i < print_n or print_n == 0:
                print("loc:", i, "src:", str(src[i]), "exp:", str(exp[i]))
            errorCnt = errorCnt + 1
        elif i < print_n:
            print("loc:", i, "src:", str(src[i]), "exp:", str(exp[i]))
    print("Is allclose:", (str(np.allclose(
        src.reshape(-1), exp.reshape(-1), atol=0.1, rtol=0.1))))
    print("Total Num:", totolNum, "error cnt:", errorCnt,
          "error percent:", float(errorCnt) / float(totolNum))
    if errorCnt > 0:
        print("compare falied")
    else:
        print("compare success")


if __name__ == "__main__":
    tik_instance = element_add_test()
    dataA = np.random.uniform(-1, 1, (1024, 256)).astype(np.float16)
    dataB = np.array([np.exp(dataA[i][j]) for i in range(
        1024) if i % 2 == 0 for j in range(256)]).astype(np.float16)
    tmp = np.array(
        [dataA[i][j] if dataA[i][j] > 0 else 0 for i in range(1024) if i % 2 == 1 for j in range(256)]).astype(
        np.float16)
    dataB = dataB.reshape(512, 256)
    tmp = tmp.reshape(512, 256)

    dataB = np.concatenate((dataB, tmp), -1)
    dataB = dataB.reshape(1024, 256)
    feed_dict = {"data_A": dataA}
    data_B, = tik_instance.tikdb.start_debug(
        feed_dict=feed_dict, interactive=False)

    dataB = dataB.reshape(-1)
    data_B = data_B.reshape(-1)

    compareData(data_B, dataB, 5)
