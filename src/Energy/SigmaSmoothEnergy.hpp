//
//  SigmaSmoothEnergy.hpp
//  OptCuts
//
//  Smoothness energy for σ1 and σ2: E = Σ_edges w_ij * ( (σ1_i - σ1_j)² + (σ2_i - σ2_j)² )
//  Minimizing |∇σ1|² + |∇σ2|² in the discrete sense (per-triangle σ, adjacent triangles).
//

#ifndef SigmaSmoothEnergy_hpp
#define SigmaSmoothEnergy_hpp

#include "Energy.hpp"

namespace OptCuts {

    class SigmaSmoothEnergy : public Energy
    {
    public:
        SigmaSmoothEnergy(void);

        virtual void getEnergyValPerElem(const TriMesh& data, Eigen::VectorXd& energyValPerElem, bool uniformWeight = false) const;
        virtual void getEnergyValByElemID(const TriMesh& data, int elemI, double& energyVal, bool uniformWeight = false) const;

        virtual void computeGradient(const TriMesh& data, Eigen::VectorXd& gradient, bool uniformWeight = false) const;

        virtual void computeHessian(const TriMesh& data, Eigen::VectorXd* V,
                                    Eigen::VectorXi* I = NULL, Eigen::VectorXi* J = NULL, bool uniformWeight = false) const;
        virtual void computeHessian(const TriMesh& data,
                                    Eigen::MatrixXd& Hessian,
                                    bool uniformWeight = false) const;

        virtual void checkEnergyVal(const TriMesh& data) const;
    };
}

#endif /* SigmaSmoothEnergy_hpp */
