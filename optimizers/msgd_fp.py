import torch
import torch.optim
from torch import optim


class MSGD_FP(optim.Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0, dampening=0, weight_decay=0, nesterov=False,
                 max_lookback=100, push_up_strategy='min', max_resolution=100, min_lookback=10, min_resolution=1,
                 quant_grads=False, results_buffer=8, lookback_momentum=0.33):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0 <= max_lookback and type(max_lookback) == int:
            raise ValueError("Invalid max_lookback value: {}".format(max_lookback))
        if not 0 <= min_lookback and type(min_lookback) == int:
            raise ValueError("Invalid max_lookback value: {}".format(min_lookback))
        if not 0 <= max_resolution and type(max_resolution) == int:
            raise ValueError("Invalid max_resolution value: {}".format(max_resolution))
        if not 0 <= min_resolution and type(min_resolution) == int:
            raise ValueError("Invalid min_resolution value: {}".format(min_resolution))
        if not push_up_strategy in ['min', 'mean', 'max'] and type(push_up_strategy) == str:
            raise ValueError("Invalid push_up_strategy value: {}".format(push_up_strategy))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, fpWeights=[])
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MSGD_FP, self).__init__(params, defaults)

        self.__make_master_cpy()
        self.__pt_kld = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.__pt_hist = torch.histc
        self.__push_up_strategy = push_up_strategy
        self.__grad_map = {}
        self.__max_lookback = max_lookback
        self.__max_resolution = max_resolution
        self.__min_lookback = min_lookback
        self.__min_resolution = min_resolution
        self.__lookback_momentum = lookback_momentum
        self.__results_buffer = results_buffer
        self.__quant_grads = quant_grads
        self.__loss_hist = []

    def __make_master_cpy(self):
        """
        Make a float32 master copy of the networks weigths for use in the backwards pass
        See: https://github.com/ICIdsl/muppet/blob/master/quant_sgd.py
        :return:
        """
        for group in self.param_groups:
            for i in range(len(group['params'])):
                group['fpWeights'].append(
                    group['params'][i].type(torch.float32).detach())
                self.__sanity_check(group['fpWeights'][-1])

    def __sanity_check(self, x):
        """
        Terminate if irrecoverable state is reached.
        :param x: object
        :return: None
        """
        assert x is not None
        if torch.is_tensor(x): assert not (torch.isnan(x).any() or torch.isinf(x).any())

    def __fpq_loss(self, w, wl, fl, quantizer, resolution):
        """
        Compute fixed point quantization loss induced by quantizing w with world length wl and fractional length fl.
        :param w: torch.Tensor, weight to be evaluated
        :param wl: int, word length
        :param fl: int, fractional length
        :param quantizer: qtorch.quant.fixed_point_quantize, quantization function
        :param resolution: int, number of bins used in EPD required by KLD
        :return: float, KLD loss induced by quantizing w
        """
        w_hat = quantizer(w, wl, fl, rounding="stochastic")
        epd, epd_hat = self.__pt_hist(w, resolution), self.__pt_hist(w_hat, resolution, torch.min(w), torch.max(w))
        epd, epd_hat = epd / torch.sum(epd), epd_hat / torch.sum(epd_hat)
        return self.__pt_kld(epd, epd_hat)

    def __g_div(self, g_list):
        """
        Compute gradient diversity of last LB gradients. Always bigger than 1 bc triangle inequality.
        :param g_list: list of torch.Tensor
        :return: float, gradient diversity
        """
        #sum_of_norms = 0
        #norm_of_sums = torch.zeros(g_list[0].shape)
        #norm_of_sums = torch.zeros(g_list[0][1].shape)
        #for g in g_list: #g_list:
        #    sum_of_norms += torch.norm(g, 2.0)
        #    norm_of_sums += g.cpu().numpy()  # This operation could be optimized
        #norm_of_sums = norm_of_sums.norm(2.0)
        norm_of_sums = g_list[0][1].norm(2.0)
        #print(sum_of_norms, flush=True)
        #return sum_of_norms / norm_of_sums# if norm_of_sums > 0 else float('inf')
        return g_list[0][0] / norm_of_sums  # if norm_of_sums > 0 else float('inf')

    def __ceil_int(self, w):
        """
        :param w: float
        :return: int
        """
        return int(w) if (int(w) - w) >= 0 else int(w) + 1

    def __push_down_wl(self, w, wl_min, wl_max, fl, quantizer, resolution):
        """
        Compute minimum fixed point quantization world length wl_min that does not cause KLD loss for a given EPD
        resolution and fl using a bisection strategy.
        :param w: torch.Tensor, weight to be evaluated
        :param wl_min: int, word length minimum to start with
        :param wl_max: int, word length maximum
        :param fl: int, fractional length
        :param quantizer: qtorch.quant.fixed_point_quantize, quantization function
        :param resolution: int, number of bins used in EPD required by KLD
        :return: int, wl_min
        """
        for n in range(wl_max - wl_min):
            wl_mid = self.__ceil_int((wl_max + wl_min) / 2)
            wl_loss = self.__fpq_loss(w, wl_mid, fl, quantizer, resolution)

            if 0 < wl_loss:
                wl_min = wl_mid
            else:
                wl_max = wl_mid
            if wl_max - wl_min <= 1:
                break
        return wl_min

    def __push_down_fl(self, w, fl_min, fl_max, wl, quantizer, resolution):
        """
        Find minimum fixed point quantization fractional length fl_min that does not cause KLD loss for a given EPD
        resolution and wl using a bisection strategy.
        :param w: torch.Tensor, weight to be evaluated
        :param fl_min: int, fractional length minimum to start with
        :param fl_max: int, fractional length maximum
        :param wl: int, world length
        :param quantizer: qtorch.quant.fixed_point_quantize, quantization function
        :param resolution: int, number of bins used in EPD required by KLD
        :return: int, fl_min
       """
        for n in range(fl_max - fl_min):
            fl_mid = self.__ceil_int((fl_max + fl_min) / 2)
            fl_loss = self.__fpq_loss(w, wl, fl_min, quantizer, resolution)
            if 0 < fl_loss:
                fl_min = fl_mid
            else:
                fl_max = fl_mid
            if fl_max - fl_min <= 1:
                break
        return fl_min

    def __push_down(self, w, fl_min, fl_mid, fl_max, wl_mid, wl_max, quantizer, resolution):
        """
        Find minimum fixed point quantization world length wl_min and fractional length fl_min
        that does not cause KLD loss for a given EPD resolution.
        :param w: torch.Tensor, weight to be evaluated
        :param fl_min: int, fractional length minimum to start with
        :param fl_mid: int, fractional length mid
        :param fl_max: int, fractional length maximum
        :param wl_mid: int, word length mid
        :param wl_max: int, word length maximum
        :param quantizer: qtorch.quant.fixed_point_quantize, quantization function
        :param resolution: int, number of bins used in EPD required by KLD
        :return: int int, wl_min fl_min
        """
        fl_min = self.__push_down_fl(w, fl_min, fl_max, wl_mid, quantizer, resolution)
        wl_min = self.__push_down_wl(w, fl_min, wl_max, fl_min, quantizer, resolution)
        return wl_min, fl_min

    def __push_up(self, g_list, wl, fl, quantizer, resolution):
        """
        Find minimum wordlength required for the network to keep learning.
        Uses GD based heuristic for quantized gradients and
        KLD based heuristic for unquantized gradients.
        :param g_list: list of torch.Tensor
        :param wl: int, word length
        :param fl: int, fractional length
        :param quantizer: torch.quant.fixed_point_quantize, quantization function
        :param resolution: int, number of bins used in EPD required by KLD
        :return: int int, wl_min fl_min
        """
        wl_min, fl_min = wl, fl
        if not self.__quant_grads:
            print('DEBUG WARNING', flush=True)
            exit(-1)
            #Deprecated
            wl_fl = []
            for g in g_list:
                fpql = self.__fpq_loss(g, wl, fl, quantizer, resolution)
                while fpql > 0 and fl <= 31 and wl <= 32:
                    if fl < wl:
                        fl += 1
                    else:
                        wl = wl + 1
                    fpql = self.__fpq_loss(g, wl, fl, quantizer, resolution)
                wl_fl.append((wl, fl))
            wl_min, fl_min = min(wl_fl)[0], min(wl_fl, key=lambda x: x[1])[1]
            if self.__push_up_strategy == 'mean':
                wl_list, fl_list = [t[0] for t in wl_fl], [t[1] for t in wl_fl]
                wl_min = sum(wl_list) / len(wl_list)
                fl_min = sum(fl_list) / len(fl_list)
                wl_min, fl_min = self.__ceil_int(wl_min), self.__ceil_int(fl_min)
            elif self.__push_up_strategy == 'max':
                wl_max, fl_max = max(wl_fl)[0], max(wl_fl, key=lambda x: x[1])[1]
                wl_min, fl_min = self.__ceil_int((wl_max + wl_min) / 2), self.__ceil_int((fl_max + fl_min) / 2)
            elif self.__push_up_strategy == 'min':
                pass  # Default case
        else:
            g, s = self.__g_div(g_list), 1
            # print(g, torch.log(torch.FloatTensor([g])), self.__ceil_int(1/(torch.log(torch.FloatTensor([g]))-1)), flush=True)
            if 0 < g < float('inf'):
                log_g_div = torch.log(torch.FloatTensor([g])) - 1
                # print(log_g_div, self.__ceil_int(1 / (log_g_div*log_g_div)),  min(self.__ceil_int(32 * (log_g_div)), 32), flush=True)
                if log_g_div > 0:
                    s1 = max(self.__ceil_int(1 / (log_g_div * log_g_div)), s)
                    s2 = max(min(self.__ceil_int(32 * (log_g_div * log_g_div)), 32) - fl_min, 0)
                    s = min(s1, s2)
                    if self.__push_up_strategy == 'mean':
                        s = self.__ceil_int(0.5 * (s1 + s2))
                    elif self.__push_up_strategy == 'max':
                        s = max(s1, s2)
                    elif self.__push_up_strategy == 'min':
                        pass

            fl_min = min(fl_min + s, 32)
            wl_min = min(max(wl_min, fl_min) + 1, 32)

        fl_min = min(fl_min, 32 - self.__results_buffer)
        wl_min = max(min(fl_min + self.__results_buffer, 32), wl_min)
        return wl_min, fl_min

    def __setstate__(self, state):
        super(MSGD_FP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def __adapt_lookback(self, g_list, lb):
        """
        Adapt lookback based on GD.
        :param g_list: list of torch.Tensor
        :param lb: int, number of gradients used for comouting GD
        :return: int, new lookback
        """
        g = self.__g_div(g_list)
        lb_new = self.__ceil_int(min(max(self.__max_lookback / g, self.__min_lookback),
                                     self.__max_lookback)) if 0 < g else self.__max_lookback
        return self.__ceil_int((lb_new * self.__lookback_momentum + lb * (1.0 - self.__lookback_momentum)))

    def __adapt_resolution(self, lb, res):
        if lb == self.__max_lookback:
            res = min(max(res + 1, self.__min_resolution), self.__max_resolution)
        if lb == self.__min_lookback:
            res = min(max(res - 1, self.__min_resolution), self.__max_resolution)
        return int(res)

    def __adapt_strategy(self, qmap, loss):
        """
        Adapt strategy used by the push up operation.
        :param qmap: dict, containing quantization mappings
        :param loss: torch.Tensor, loss after last fwd pass or accumulated loss in case of accum. gradients
        :return: None
        """
        avg_lb = int(sum(qmap['lb']) / len(qmap['lb']))
        if len(self.__loss_hist) > avg_lb:
            avg_loss_abs = torch.abs(sum(self.__loss_hist) / avg_lb)
            loss_abs = torch.abs(loss)
            if avg_loss_abs <= loss_abs and self.__push_up_strategy == 'mean': self.__push_up_strategy = 'max'
            if avg_loss_abs <= loss_abs and self.__push_up_strategy == 'min': self.__push_up_strategy = 'mean'
            if avg_loss_abs > loss_abs: self.__push_up_strategy = 'min'
            self.__loss_hist = self.__loss_hist[-avg_lb:]
        self.__loss_hist.append(loss)

    def __debug_print(self, qmap):
        """
        :param qmap: dict, containing quantization mappings
        :return: None
        """
        avg_lb = int(sum(qmap['lb']) / len(qmap['lb']))
        avg_wl = int(sum(qmap['qwl']) / len(qmap['qwl']))
        avg_fl = int(sum(qmap['qfl']) / len(qmap['qfl']))
        avg_res = int(sum(qmap['res']) / len(qmap['res']))
        print('PU:', self.__push_up_strategy, flush=True)
        print('WL:', qmap['qwl'], 'AVG_WL:', avg_wl, flush=True)
        print('FL:', qmap['qfl'], 'AVG_FL:', avg_fl, flush=True)
        print('RES:', qmap['res'], 'AVG_RES:', avg_res, flush=True)
        print('LB:', qmap['lb'], 'AVG_LB:', avg_lb, "\n", flush=True)

    @torch.no_grad()
    def step(self, qmap=None, closure=None):
        """
        Performs a single optimization step.
        :param qmap: dict
        :param closure: closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        :return: loss
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.__adapt_strategy(qmap, loss)

        for group in self.param_groups:
            params_with_grad = []
            params_with_grad_quantized = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for i, p in enumerate(group['params']):
                # Marvin
                if p.grad is not None:
                    # Quantize grads
                    if self.__quant_grads:
                        p.grad.data = qmap['q'][i](
                            p.grad.data,
                            wl=qmap['qwl'][i], fl=qmap['qfl'][i],
                            rounding="stochastic")

                    # Normalize Grads
                    #torch.nn.utils.clip_grad_norm_(p.grad.data, 1.0, norm_type=2.0)

                    #print('A',torch.norm(p.grad.data, 2.0), flush=True)
                    n = torch.norm(p.grad.data, 2.0)
                    if n > 0.0:
                        p.grad.data = p.grad.data/n
                    #print(n, torch.norm(p.grad.data, 2.0))
                    #print('B',torch.norm(p.grad.data, 2.0), flush=True)
                    #print(float(n > 0.0), flush=True)
                    # Part 1 of SGD
                    params_with_grad.append(group['fpWeights'][i])  # fpWeights
                    params_with_grad_quantized.append(p)
                    d_p_list.append(p.grad.data)
                    state = self.state[group['params'][i]]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
                    # Part 1 of SGD
                    # If we look at a weight, check WL, FL, RES, LB
                    if len(p.data.shape) > 1:
                        if i in self.__grad_map:
                            #print(d_p_list[-1].shape, flush=True)
                            #self.__grad_map[i].append(p.grad.data)
                            self.__grad_map[i][0] = (self.__grad_map[i][0][0]+1, self.__grad_map[i][0][1] + p.grad.data)
                        else:
                            self.__grad_map[i] = []
                            #self.__grad_map[i].append(p.grad.data)
                            self.__grad_map[i].append((1, p.grad.data))

                        # Adapt LB before tesing conditions to avoid throwing away gradients we could still use
                        # Adapt RES
                        qmap['lb'][i] = self.__adapt_lookback(self.__grad_map[i], qmap['lb'][i])
                        qmap['res'][i] = self.__adapt_resolution(qmap['lb'][i], qmap['res'][i])

                        # Update wl, fl based on their KLD
                        fl_min, fl_mid, fl_max = 4, qmap['qfl'][i], qmap['qfl'][i]
                        wl_min, wl_mid, wl_max = fl_max, qmap['qwl'][i], qmap['qwl'][i]
                        #print(self.__grad_map[i][0][0], qmap['lb'][i], flush=True)
                        #if len(self.__grad_map[i]) >= qmap['lb'][i]:
                        if i in self.__grad_map and self.__grad_map[i][0][0] >= qmap['lb'][i]:
                            wl_min, fl_min = self.__push_down(group['fpWeights'][i][0], fl_min, fl_mid, fl_max, wl_mid,
                                                              wl_max, qmap['q'][i], qmap['res'][i])
                            wl_min, fl_min = self.__push_up(self.__grad_map[i], wl_min, fl_min, qmap['q'][i],
                                                            qmap['res'][i])
                            #self.__grad_map[i] = []
                            #print('PM', flush=True)
                            self.__grad_map.pop(i)
                            #print(i in self.__grad_map, flush=True)
                            qmap['qwl'][i], qmap['qfl'][i] = wl_min, fl_min

                    else:
                        # If we are looking at a bias, use the values from previous weight
                        qmap['qwl'][i], qmap['qfl'][i], qmap['res'][i], qmap['lb'][i] = qmap['qwl'][i - 1], qmap['qfl'][
                            i - 1], qmap['res'][i - 1], qmap['lb'][i - 1]

            # Part 2 of SGD
            j = 0
            for i, param in enumerate(group['params']):
                if param.grad is not None:
                    d_p = d_p_list[j]
                    if weight_decay != 0:
                        d_p = d_p.add(params_with_grad[j], alpha=weight_decay)
                    if momentum != 0:
                        buf = momentum_buffer_list[j]
                        if buf is None:
                            buf = torch.clone(d_p).detach()
                            momentum_buffer_list[j] = buf
                        else:
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf
                    params_with_grad[j].add_(d_p, alpha=-lr)

                    # Quantize updated weights and write them back to quantized weights
                    # s.t. they are read for the next fwd pass
                    group['params'][i].data = qmap['q'][i](
                        params_with_grad[j].data,
                        wl=qmap['qwl'][i], fl=qmap['qfl'][i],
                        rounding="stochastic")

                    qmap['sp'][i] = (group['params'][i].data.numel() - group['params'][i].data.nonzero().size(0))/group['params'][i].data.numel()

                    j += 1

            for p, momentum_buffer in zip(params_with_grad_quantized, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
        return loss
